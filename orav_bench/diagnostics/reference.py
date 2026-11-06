"""Face identity, style features, and foreground motion direction."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .common import frames


class Face:
    def __init__(self, device, assets):
        self.device = device
        self.assets = assets
        torch.set_num_threads(2)
        torch.manual_seed(0)
        from retinaface.network import RetinaFace

        self.det = RetinaFace(
            name="Resnet50",
            pretrained=False,
            in_channels=256,
            return_layers={"layer2": 1, "layer3": 2, "layer4": 3},
            out_channels=256,
        )
        state = torch.load(assets["retinaface"], map_location="cpu", weights_only=True)
        self.det.load_state_dict(state)
        self.det.to(self.device).eval()
        spec = importlib.util.spec_from_file_location("arcface_orig", assets["arcface_source"])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.enc = mod.resnet_face18(use_se=False)
        state = torch.load(assets["arcface"], map_location="cpu", weights_only=True)
        self.enc.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        self.enc.to(self.device).eval()
        print("face_models_loaded", flush=True)

    @torch.inference_mode()
    def detect(self, im):
        from retinaface.box_utils import decode
        from retinaface.prior_box import priorbox

        w, h = im.size
        scale = 960 / max(w, h)
        size = (round(w * scale), round(h * scale))
        small = im.resize(size, Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(small).copy()).permute(2, 0, 1).float().to(self.device) / 255
        x = (x - torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225], device=self.device
        )[:, None, None]
        loc, conf, land = self.det(x[None])
        scores = conf[0].softmax(-1)[:, 1]
        priors = priorbox(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
            image_size=(size[1], size[0]),
        ).to(self.device)
        boxes = decode(loc[0], priors, [0.1, 0.2]) * torch.tensor([w, h, w, h], device=self.device)
        chosen = scores > 0.7
        boxes = boxes[chosen].cpu()
        scores = scores[chosen].cpu()
        indices = nms(boxes, scores, 0.4).tolist()[:16]
        out = []
        for i in indices:
            box = boxes[i].numpy()
            box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
            if min(box[2] - box[0], box[3] - box[1]) < 12:
                continue
            out.append({"box": box.tolist(), "confidence": float(scores[i])})
        return out

    @torch.inference_mode()
    def features(self, im, det):
        if not det:
            return np.empty((0, 1024))
        imgs = []
        for d in det:
            x = np.asarray(
                im.crop(tuple(d["box"])).resize((128, 128), Image.Resampling.BILINEAR).convert("L")
            ).astype("float32")
            imgs.extend([(x - 127.5) / 127.5, (x[:, ::-1].copy() - 127.5) / 127.5])
        z = self.enc(torch.from_numpy(np.stack(imgs))[:, None].to(self.device)).reshape(len(det), -1)
        return F.normalize(z, dim=-1).cpu().numpy()


def similarity(a, b, ma, mb):
    # [time, tracks, xy] -> correlation of unit step directions.
    da = np.diff(a, axis=0)
    db = np.diff(b, axis=0)
    na = np.linalg.norm(da, axis=-1)
    nb = np.linalg.norm(db, axis=-1)
    va = ma[:-1] & ma[1:] & (na >= 0.5)
    vb = mb[:-1] & mb[1:] & (nb >= 0.5)
    ua = da / np.maximum(na[..., None], 1e-12)
    ub = db / np.maximum(nb[..., None], 1e-12)
    common = va[:, :, None] & vb[:, None, :]
    count = common.sum(0)
    dot = np.einsum("tnd,tmd->tnm", ua, ub)
    sim = np.where(count >= 8, (dot * common).sum(0) / np.maximum(count, 1), np.nan)
    row = np.any(np.isfinite(sim), axis=1)
    col = np.any(np.isfinite(sim), axis=0)
    value = None
    if row.any() and col.any():
        value = float((np.nanmax(sim[row], axis=1).mean() + np.nanmax(sim[:, col], axis=0).mean()) / 2)
    return {
        "bidirectional_direction_cosine": value,
        "source_tracks": a.shape[1],
        "output_tracks": b.shape[1],
        "source_comparable_tracks": int(row.sum()),
        "output_comparable_tracks": int(col.sum()),
        "source_moving_visible_fraction": float(va.mean()) if va.size else None,
        "output_moving_visible_fraction": float(vb.mean()) if vb.size else None,
    }


class Motion:
    def __init__(self, device, assets):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        self.device = device
        self.assets = assets
        src = Path(assets["cotracker_source"])
        sys.path.insert(0, str(src))
        from cotracker.predictor import CoTrackerPredictor

        self.model = (
            CoTrackerPredictor(checkpoint=str(assets["cotracker"]), offline=True, v2=False, window_len=60)
            .to(self.device)
            .eval()
        )
        root = assets["grounding_dino"]
        self.processor = AutoProcessor.from_pretrained(root, local_files_only=True)
        self.detector = (
            AutoModelForZeroShotObjectDetection.from_pretrained(root, local_files_only=True)
            .to(self.device)
            .eval()
        )
        self.detector.config.disable_custom_kernels = True
        print("motion_models_loaded", flush=True)

    @torch.inference_mode()
    def media(self, r):
        ims, ts = frames(r, n=32, size=320)
        arr = np.stack([np.asarray(im) for im in ims])
        video = torch.from_numpy(arr.copy()).permute(0, 3, 1, 2)[None].float().to(self.device)
        tracks, visible = self.model(video, grid_size=16, grid_query_frame=0, backward_tracking=False)
        tracks = tracks[0].cpu().numpy()
        visible = visible[0].cpu().numpy().astype(bool)
        inp = self.processor(images=ims[0], text="person . animal . vehicle .", return_tensors="pt").to(
            self.device
        )
        boxes = (
            self.processor.post_process_grounded_object_detection(
                self.detector(**inp),
                inp.input_ids,
                threshold=0.25,
                text_threshold=0.25,
                target_sizes=[(ims[0].height, ims[0].width)],
            )[0]["boxes"]
            .cpu()
            .numpy()
        )
        xy = tracks[0]
        fg = np.zeros(len(xy), bool)
        for x1, y1, x2, y2 in boxes:
            fg |= (xy[:, 0] >= x1) & (xy[:, 0] <= x2) & (xy[:, 1] >= y1) & (xy[:, 1] <= y2)
        return {
            "tracks": tracks,
            "visible": visible,
            "fg": fg,
            "timestamps": ts,
            "boxes": boxes.tolist(),
            "size": [ims[0].width, ims[0].height],
        }
