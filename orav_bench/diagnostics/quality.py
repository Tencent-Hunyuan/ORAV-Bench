"""Published quality estimators; model implementations are external dependencies."""

from __future__ import annotations

import hashlib
import importlib.util
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import audio


@contextmanager
def trusted_checkpoint_loading():
    """Compatibility for trusted VBench checkpoints on PyTorch 2.6 and later."""
    loader = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return loader(*args, **kwargs)

    try:
        torch.load = load
        yield
    finally:
        torch.load = loader


def native_frames(record, count=8):
    import decord

    decord.bridge.set_bridge("native")
    reader = decord.VideoReader(record["path"], num_threads=2)
    indices = np.minimum(len(reader) - 1, ((np.arange(count) + 0.5) * len(reader) / count).astype(int))
    images = [Image.fromarray(x) for x in reader.get_batch(indices.tolist()).asnumpy()]
    return images, {
        "indices": indices.tolist(),
        "timestamps": reader.get_frame_timestamp(indices.tolist()).tolist(),
    }


class Dover:
    def __init__(self, device, assets):
        import yaml

        sys.path.insert(0, str(assets["dover_source"]))
        import dover.models.evaluator as evaluator
        from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition

        self.device, self.decompose = device, spatial_temporal_view_decomposition
        options = yaml.safe_load((Path(assets["dover_source"]) / "dover.yml").read_text())
        factory = evaluator.convnext_3d_tiny
        try:
            evaluator.convnext_3d_tiny = lambda **kw: factory(**{**kw, "pretrained": False})
            self.model = evaluator.DOVER(**options["model"]["args"])
        finally:
            evaluator.convnext_3d_tiny = factory
        checkpoint = torch.load(assets["dover"], map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
        self.model.to(device).eval()
        self.samples = options["data"]["val-l1080p"]["args"]["sample_types"]
        self.samplers = {
            name: UnifiedFrameSampler(
                s["clip_len"] // s["t_frag"], s["t_frag"], s["frame_interval"], s["num_clips"]
            )
            if "t_frag" in s
            else UnifiedFrameSampler(s["clip_len"], s["num_clips"], s["frame_interval"])
            for name, s in self.samples.items()
        }

    @torch.inference_mode()
    def score(self, record):
        # Content-derived seeds make sharding and execution order irrelevant.
        seed = int(hashlib.sha256(("orav-quality:" + record["sha256"]).encode()).hexdigest()[:8], 16)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        views, indices = self.decompose(record["path"], self.samples, self.samplers)
        mean, std = torch.tensor([123.675, 116.28, 103.53]), torch.tensor([58.395, 57.12, 57.375])
        for key, value in views.items():
            clips = self.samples[key]["num_clips"]
            views[key] = (
                ((value.permute(1, 2, 3, 0) - mean) / std)
                .permute(3, 0, 1, 2)
                .reshape(value.shape[0], clips, -1, *value.shape[2:])
                .transpose(0, 1)
                .to(self.device)
            )
        values = self.model(views)
        return {
            "scores": {"dover_" + key: float(value.mean()) for key, value in zip(views, values)},
            "seed": seed,
            "frame_indices": {key: value.tolist() for key, value in indices.items()},
        }


class Aesthetic:
    def __init__(self, device, assets):
        self.device = device
        source = Path(assets["aesthetic_source"])
        sys.path.insert(0, str(source))
        paths = list(source.rglob("siglip_v2_5.py"))
        if len(paths) != 1:
            raise ValueError("Provide the Aesthetic Predictor V2.5 source checkout")
        spec = importlib.util.spec_from_file_location("aesthetic_v25", paths[0])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model, self.processor = module.convert_v2_5_from_siglip(
            predictor_name_or_path=str(assets["aesthetic_head"]),
            encoder_model_name=str(assets["siglip"]),
            local_files_only=True,
        )
        self.model.to(device).eval()

    @torch.inference_mode()
    def score(self, record):
        images, frames = native_frames(record)
        values = (
            self.model(**self.processor(images=images, return_tensors="pt").to(self.device))
            .logits.flatten()
            .float()
            .cpu()
            .numpy()
        )
        return {
            "scores": {"aesthetic_v25": float(values.mean())},
            "frames": frames,
            "per_frame": values.tolist(),
        }


class Audiobox:
    def __init__(self, device, assets):
        from audiobox_aesthetics.infer import AesPredictor
        from audiobox_aesthetics.model.aes import AesMultiOutput, Normalize

        model = AesMultiOutput.from_pretrained(str(assets["audiobox"])).to(device).eval()
        self.predictor = object.__new__(AesPredictor)
        self.predictor.model, self.predictor.device = model, torch.device(device)
        self.predictor.sample_rate, self.predictor.data_col = 16000, "path"
        self.predictor.target_transform = {
            axis: Normalize(**model.target_transform[axis]) for axis in ("CE", "CU", "PC", "PQ")
        }

    def score(self, record):
        samples, meta = audio(record, 16000, True)
        if samples is None or meta["state"] != "signal":
            return {"scores": {}, "audio": meta}
        value = self.predictor.forward(
            [{"path": torch.from_numpy(samples.mean(0, keepdims=True)), "sample_rate": 16000}]
        )[0]
        return {"scores": {"audiobox_pq": float(value["PQ"])}, "audio": meta}


class VBench:
    """The five published dimensions used to compute the corpus-standardized control."""

    def __init__(self, device, assets):
        import clip
        import vbench
        from pyiqa.archs.musiq_arch import MUSIQ
        from vbench.aesthetic_quality import get_aesthetic_model
        from vbench.motion_smoothness import MotionSmoothness

        from .visual import Encoder

        self.device = device
        # Some official checkpoints require the trusted-checkpoint loader used by the packages.
        self.dino = Encoder(device, assets).dino
        self.clip_b32, self.preprocess_b32 = clip.load(str(assets["clip_b32"]), device=device)
        self.clip_l14, _ = clip.load(str(assets["clip_l14"]), device=device)
        with trusted_checkpoint_loading():
            self.aesthetic = get_aesthetic_model(str(assets["laion_aesthetic"])).to(device)
            self.motion = MotionSmoothness(
                str(Path(vbench.__file__).parent / "third_party/amt/cfgs/AMT-S.yaml"),
                str(assets["amt"]),
                device,
            )
            self.musiq = MUSIQ(pretrained_model_path=str(assets["musiq"])).to(device).eval()

    def score(self, record):
        from vbench.aesthetic_quality import laion_aesthetic
        from vbench.background_consistency import background_consistency
        from vbench.imaging_quality import technical_quality
        from vbench.motion_smoothness import motion_smoothness
        from vbench.subject_consistency import subject_consistency

        one = [record["path"]]
        calls = {
            "subject_consistency": lambda: subject_consistency(self.dino, one, self.device, False),
            "background_consistency": lambda: background_consistency(
                self.clip_b32, self.preprocess_b32, one, self.device, False
            ),
            "motion_smoothness": lambda: motion_smoothness(self.motion, one),
            "aesthetic_quality": lambda: laion_aesthetic(self.aesthetic, self.clip_l14, one, self.device),
            "imaging_quality": lambda: technical_quality(self.musiq, one, self.device),
        }
        scores, errors = {}, {}
        for name, call in calls.items():
            try:
                _, rows = call()
                scores["vbench_" + name] = float(rows[0]["video_results"])
            except Exception as error:
                errors[name] = str(error)
        return {"scores": scores, "errors": errors}
