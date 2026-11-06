"""Full-frame CLIP and DINO reference affinity."""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .common import frames


def tensor(im, mean, std):
    # Preserve the entire selected region, its aspect ratio, and equal 224-pixel scale.
    im = ImageOps.contain(im, (224, 224), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (224, 224), tuple(round(x * 255) for x in mean))
    canvas.paste(im, ((224 - im.width) // 2, (224 - im.height) // 2))
    a = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255
    return (a - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]


class Encoder:
    def __init__(self, device, assets):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        self.device = device
        self.assets = assets
        from open_clip.openai import load_openai_model

        self.clip = load_openai_model(str(assets["clip_b32"]), precision="fp32", device=self.device).eval()
        sys.path.insert(0, str(assets["dino_source"]))
        import vision_transformer

        self.dino = vision_transformer.vit_base(patch_size=16, num_classes=0)
        self.dino.load_state_dict(torch.load(assets["dino_b16"], map_location="cpu", weights_only=True))
        self.dino.to(self.device).eval()
        self.cache = {}
        print("models_loaded", flush=True)

    @torch.inference_mode()
    def encode(self, ims):
        flat = list(ims)
        result = {}
        for name, model, mean, std in [
            ("clip", self.clip, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ("dino", self.dino, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]:
            chunks = []
            for i in range(0, len(flat), 32):
                x = torch.stack([tensor(im, mean, std) for im in flat[i : i + 32]]).to(self.device)
                y = model.encode_image(x) if name == "clip" else model(x)
                chunks.append(F.normalize(y.float(), dim=-1).cpu().numpy())
            result[name] = np.concatenate(chunks).reshape(len(ims), -1)
        return result

    def media(self, r):
        h = r["sha256"]
        if h in self.cache:
            return self.cache[h]
        ims, ts = frames(r)
        e = self.encode(ims)
        self.cache[h] = (e, ts)
        return e, ts
