"""Within-category region matching and foreground-masked scene affinity."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.ops import nms
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Config,
    Sam2Model,
    Sam2Processor,
)

ANIMALS = [
    "cat",
    "dog",
    "horse",
    "rabbit",
    "bird",
    "duck",
    "chicken",
    "lion",
    "tiger",
    "elephant",
    "monkey",
    "panda",
    "robot",
    "bear",
    "deer",
    "fish",
    "dolphin",
]
PROPS = [
    "sphygmomanometer",
    "basketball",
    "soccer ball",
    "football",
    "toy ball",
    "medicine ball",
    "tennis ball",
    "tennis racket",
    "badminton racket",
    "microphone",
    "stethoscope",
    "surfboard",
    "snowboard binding",
    "snowboard",
    "ice skates",
    "figure skates",
    "skating costume",
    "costume",
    "judogi",
    "swimsuit",
    "long dress",
    "gown",
    "cape",
    "skirt",
    "shoes",
    "shoe",
    "sneakers",
    "footwear",
    "scarf",
    "bucket hat",
    "hat",
    "baseball cap",
    "fan",
    "sculpture",
    "diabolo",
    "yo-yo",
    "metronome",
    "bell",
    "pommel horse",
    "barbell",
    "hoop",
    "gymnastic rings",
    "gymnastics rings",
    "training ring",
    "suspended bar",
    "vaulting barrel",
    "juggling clubs",
    "club",
    "ribbon",
    "whip",
    "cane",
    "jump rope",
    "lead rope",
    "rope",
    "knife",
    "spoon",
    "sourdough bread",
    "bread",
    "jar",
    "tomatoes",
    "french press",
    "kettle",
    "cooking pot",
    "skillet",
    "mixing bowl",
    "mug",
    "handbag",
    "tote bag",
    "bag",
    "suitcase",
    "clipboard",
    "marker pen",
    "paper map",
    "calendar",
    "cabinet",
    "file cart",
    "crate",
    "basket",
    "watering can",
    "rake",
    "pruning shears",
    "garden hose",
    "horse brush",
    "hot air balloon",
    "kite",
    "bouquet",
    "boxing gloves",
    "focus mitts",
    "tuning fork",
    "rubber mallet",
    "mat",
    "hand drum",
    "towel",
    "tape roll",
    "wristwatch",
    "bmx bike",
    "bicycle",
    "motorcycle",
]
ALIASES = {
    "football": "soccer ball",
    "figure skates": "ice skates",
    "shoes": "shoe",
    "sneakers": "shoe",
    "footwear": "shoe",
    "skating costume": "costume",
    "long dress": "dress",
    "gown": "dress",
    "judogi": "judo uniform",
    "bucket hat": "hat",
    "baseball cap": "hat",
    "gymnastics rings": "gymnastic rings",
    "club": "juggling club",
    "juggling clubs": "juggling club",
    "bmx bike": "bicycle",
    "toy ball": "ball",
    "sourdough bread": "bread",
}


def categories(r):
    s = r["scope"].lower()
    if r["semantic"] == "subject":
        if re.search(r"\b(man|woman|person|people|dancer|performer|athlete|boy|girl)\b", s):
            return ["person"]
        words = [w for w in ANIMALS if re.search(r"\b" + re.escape(w) + r"s?\b", s)]
        return words or ["person"]
    found = []
    for word in sorted(PROPS, key=len, reverse=True):
        if word in s and not any(word in old for old in found):
            found.append(word)
    return sorted(set(ALIASES.get(w, w) for w in found))


def prepare(im, mean, std):
    im = ImageOps.contain(im, (224, 224), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (224, 224), tuple(round(x * 255) for x in mean))
    canvas.paste(im, ((224 - im.width) // 2, (224 - im.height) // 2))
    x = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255
    return (x - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]


class RegionModels:
    def __init__(self, device, assets):
        self.device = device
        self.assets = assets
        torch.set_num_threads(2)
        torch.manual_seed(0)
        d = assets["grounding_dino"]
        self.processor = AutoProcessor.from_pretrained(d, local_files_only=True)
        self.detector = (
            AutoModelForZeroShotObjectDetection.from_pretrained(d, local_files_only=True)
            .to(self.device)
            .eval()
        )
        self.detector.config.disable_custom_kernels = True
        s = Path(assets["sam2"])
        cfg = json.loads((s / "config.json").read_text())
        cfg["model_type"] = "sam2"
        cfg["architectures"] = ["Sam2Model"]
        self.segmenter, load_info = Sam2Model.from_pretrained(
            s, config=Sam2Config.from_dict(cfg), local_files_only=True, output_loading_info=True
        )
        assert not load_info["missing_keys"] and not load_info["mismatched_keys"], load_info
        self.segmenter.to(self.device).eval()
        self.sam_processor = Sam2Processor.from_pretrained(s, local_files_only=True)
        self.sam_loading_info = load_info
        from open_clip.openai import load_openai_model

        self.clip = load_openai_model(str(assets["clip_b32"]), precision="fp32", device=self.device).eval()
        sys.path.insert(0, str(assets["dino_source"]))
        import vision_transformer

        self.dino = vision_transformer.vit_base(patch_size=16, num_classes=0)
        self.dino.load_state_dict(torch.load(assets["dino_b16"], map_location="cpu", weights_only=True))
        self.dino.to(self.device).eval()
        print("region_models_loaded", flush=True)

    @torch.inference_mode()
    def detect(self, im, labels):
        if not labels:
            return []
        text = " . ".join(labels) + " ."
        inputs = self.processor(images=im, text=text, return_tensors="pt").to(self.device)
        outputs = self.detector(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.25,
            text_threshold=0.25,
            target_sizes=[(im.height, im.width)],
        )[0]
        boxes = result["boxes"]
        scores = result["scores"]
        names = result.get("text_labels", result.get("labels"))
        keep = nms(boxes.cpu(), scores.cpu(), 0.5).tolist()
        items = []
        for i in keep[:32]:
            x1, y1, x2, y2 = boxes[i].tolist()
            box = [max(0, x1), max(0, y1), min(im.width, x2), min(im.height, y2)]
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            items.append({"box": box, "confidence": float(scores[i]), "label": str(names[i])})
        return items

    @torch.inference_mode()
    def embeddings(self, ims):
        if not ims:
            return {}
        result = {}
        for name, model, mean, std in [
            ("clip", self.clip, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ("dino", self.dino, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]:
            chunks = []
            for start in range(0, len(ims), 32):
                x = torch.stack([prepare(im, mean, std) for im in ims[start : start + 32]]).to(self.device)
                y = model.encode_image(x) if name == "clip" else model(x)
                chunks.append(F.normalize(y.float(), dim=-1).cpu().numpy())
            result[name] = np.concatenate(chunks)
        return result

    @torch.inference_mode()
    def background(self, im, detections):
        if not detections:
            return None, {"state": "no_detected_foreground"}
        inputs = self.sam_processor(
            images=im, input_boxes=[[d["box"] for d in detections]], return_tensors="pt"
        ).to(self.device)
        out = self.segmenter(
            **{k: v for k, v in inputs.items() if k != "original_sizes"}, multimask_output=False
        )
        masks = self.sam_processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        union = masks.reshape(-1, im.height, im.width).any(dim=0).numpy()
        frac = float(union.mean())
        if frac > 0.9:
            return None, {"state": "insufficient_background", "foreground_fraction": frac}
        arr = np.asarray(im).copy()
        arr[union] = [123, 117, 104]
        return Image.fromarray(arr), {
            "state": "masked",
            "foreground_fraction": frac,
            "method": "SAM2.1 box-conditioned foreground union; constant fill RGB123/117/104",
        }


def match_allowed(label, allowed):
    return any(a in label or label in a for a in allowed)
