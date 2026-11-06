"""Task-aware adapters for the paper's independently interpretable diagnostics."""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

from .common import audio, frames, probe


def load_model(family, device, assets):
    if family == "visual":
        from .visual import Encoder

        return Encoder(device, assets)
    if family in ("speech", "music"):
        from .audio import Audio

        primary = Audio(device, assets, role=family)
        if family == "speech":
            return primary
        from .audio_models import Music

        return primary, Music(device, assets)
    if family == "nisqa":
        from .audio_models import Nisqa

        return Nisqa(device, assets)
    if family in ("face", "motion"):
        from .reference import Face, Motion

        return {"face": Face, "motion": Motion}[family](device, assets)
    if family == "localization":
        from .localization import RegionModels

        return RegionModels(device, assets)
    if family in ("vbench", "dover", "aesthetic", "audiobox"):
        from .quality import Aesthetic, Audiobox, Dover, VBench

        return {"vbench": VBench, "dover": Dover, "aesthetic": Aesthetic, "audiobox": Audiobox}[family](
            device, assets
        )
    if family == "style":
        return Style(device, assets)
    raise ValueError(f"Unknown diagnostic family: {family}")


class Style:
    def __init__(self, device, assets):
        sys.path.insert(0, str(assets["aladin_source"]))
        from main import AViT

        self.device = device
        self.model = AViT()
        self.model.load_state_dict(torch.load(assets["aladin"], map_location="cpu", weights_only=True))
        self.model.to(device).eval()

    @torch.inference_mode()
    def encode(self, images):
        x = torch.stack(
            [
                torch.from_numpy(np.asarray(im.resize((224, 224), Image.Resampling.BILINEAR)).copy())
                .permute(2, 0, 1)
                .float()
                / 255
                for im in images
            ]
        )
        return (
            torch.nn.functional.normalize(
                self.model.encode(((x - 0.5) / 0.5).to(self.device)).float(), dim=-1
            )
            .cpu()
            .numpy()
        )


def role_means(task, per_reference):
    """Require every assigned reference of a role, rather than silently changing its scope."""
    output = {}
    for role in {r["semantic"] for r in task["references"]}:
        handles = [r["handle"] for r in task["references"] if r["semantic"] == role]
        metrics = {name for handle in handles for name in per_reference.get(handle, {})}
        for name in metrics:
            values = [per_reference.get(handle, {}).get(name) for handle in handles]
            if all(value is not None and np.isfinite(value) for value in values):
                output[f"{role}_{name}"] = float(np.mean(values))
    return output


def measure(family, model, task, candidate):
    if family in ("vbench", "dover", "aesthetic", "audiobox"):
        return model.score(candidate)
    if family == "nisqa":
        streams = probe(candidate["path"])["streams"]
        stream = next((s for s in streams if s["codec_type"] == "audio"), None)
        if stream is None:
            return {"scores": {}, "state": "no_audio_stream"}
        sr = int(model.args["ms_sr"] or stream["sample_rate"])
        samples, meta = audio(candidate, sr)
        scores = model.score(samples[0], sr) if samples is not None and meta["state"] == "signal" else {}
        return {"scores": {"nisqa_" + k: value for k, value in scores.items() if k == "mos"}, "audio": meta}
    if family == "localization":
        return localized(model, task, candidate)
    if family == "face":
        return face(model, task, candidate)
    per_reference, details = {}, {}
    if family == "visual":
        observed, times = model.media(candidate)
        for ref in task["references"]:
            if ref["modality"] == "audio":
                continue
            target, _ = model.media(ref)
            per_reference[ref["handle"]] = {
                name: float((observed[name] @ target[name].T).mean()) for name in observed
            }
        details["timestamps"] = times
    elif family == "style":
        observed = model.encode(frames(candidate)[0])
        for ref in task["references"]:
            if ref["semantic"] == "style":
                per_reference[ref["handle"]] = {
                    "aladin": float((observed @ model.encode(frames(ref)[0]).T).mean())
                }
    elif family == "motion":
        from .reference import similarity

        observed = model.media(candidate)
        for ref in task["references"]:
            if ref["semantic"] != "motion":
                continue
            target = model.media(ref)
            a, b = target["fg"], observed["fg"]
            score = None
            if target["boxes"] and observed["boxes"] and a.sum() >= 3 and b.sum() >= 3:
                score = similarity(
                    target["tracks"][:, a],
                    observed["tracks"][:, b],
                    target["visible"][:, a],
                    observed["visible"][:, b],
                )["bidirectional_direction_cosine"]
            per_reference[ref["handle"]] = {"direction": score}
        details["timestamps"] = observed["timestamps"]
    else:
        from .audio import error_rate, waveform

        primary, music = model if family == "music" else (model, None)
        role = "music" if music else "speech"
        observed = primary.media(candidate, role)
        details["audio"] = observed["result"]
        for ref in task["references"]:
            if ref["semantic"] != role:
                continue
            target = primary.media(ref, role)
            values = {}
            reuse = waveform(target["waveform"], observed["waveform"])
            if reuse is not None:
                values["waveform"] = reuse["abs_cosine"]
            if role == "speech":
                if target["voice"] is not None and observed["voice"] is not None:
                    values["voice"] = float(target["voice"] @ observed["voice"])
            else:
                if target["music"] is not None and observed["music"] is not None:
                    values["clap"] = float((observed["music"] @ target["music"].T).mean())
                first, fm = audio(ref, 24000)
                second, sm = audio(candidate, 24000)
                if first is not None and second is not None and fm["state"] == sm["state"] == "signal":
                    u, a = music.encode(first[0])
                    v, b = music.encode(second[0])
                    values["muq"] = float(u @ v)
                    ca, cb = np.asarray(a["chroma_frames"]), np.asarray(b["chroma_frames"])
                    ca, cb = ca[:, np.linalg.norm(ca, axis=0) > 0.5], cb[:, np.linalg.norm(cb, axis=0) > 0.5]
                    if min(ca.shape[1], cb.shape[1]) >= 2:
                        import librosa

                        cost = np.clip(1 - ca.T @ cb, 0, 2)
                        _, path = librosa.sequence.dtw(C=cost)
                        values["chroma"] = float(1 - cost[path[:, 0], path[:, 1]].mean())
            per_reference[ref["handle"]] = values
        if role == "speech":
            transcript = observed["result"].get("transcript")
            target = task["diagnostics"].get("speech_target")
            # No target text is supplied to ASR. Only the subsequent edit distance uses it.
            if target and transcript is not None:
                details["text_error"] = error_rate(target, transcript)
            mos = observed["result"].get("dnsmos_longest_speech_segment")
            details["dnsmos"] = mos
    scores = role_means(task, per_reference)
    if details.get("text_error"):
        scores[details["text_error"]["metric"].lower()] = details["text_error"]["value"]
    if details.get("dnsmos"):
        scores["dnsmos_ovrl"] = details["dnsmos"]["OVRL"]
    return {"scores": scores, "per_reference": per_reference, **details}


def face(model, task, candidate):
    targets, handles = [], []
    for ref in task["references"]:
        if ref["semantic"] != "subject":
            continue
        image = frames(ref)[0][0]
        detections = model.detect(image)
        if len(detections) != 1:
            continue
        targets.append(model.features(image, detections)[0])
        handles.append(ref["handle"])
    if not targets:
        return {"scores": {}, "state": "no_unambiguous_reference_face"}
    samples = {handle: [] for handle in handles}
    for image in frames(candidate)[0]:
        features = model.features(image, model.detect(image))
        if not len(features):
            continue
        similarity = np.stack(targets) @ features.T
        ii, jj = linear_sum_assignment(-similarity)
        for i, j in zip(ii, jj):
            samples[handles[i]].append(float(similarity[i, j]))
    valid = [float(np.mean(values)) for values in samples.values() if values]
    return {
        "scores": {"face_cosine": float(np.mean(valid))} if len(valid) == len(handles) else {},
        "per_reference": samples,
        "measurable_reference_handles": handles,
    }


def localized(model, task, candidate):
    from .localization import categories, match_allowed

    refs = [
        {**r, "scope": task["diagnostics"].get("reference_scope", {}).get(r["handle"], "")}
        for r in task["references"]
        if r["semantic"] in ("subject", "prop")
    ]
    if any(not r["scope"] for r in refs):
        raise ValueError("Localization requires diagnostics.reference_scope for subject/prop references")
    labels = sorted({category for ref in refs for category in categories(ref)})
    targets = {}
    for ref in refs:
        crops = []
        for image in frames(ref)[0]:
            detections = model.detect(image, categories(ref))
            if detections:
                crops.append(image.crop(tuple(max(detections, key=lambda d: d["confidence"])["box"])))
        targets[ref["handle"]] = model.embeddings(crops)
    bg_case = {"scene", "motion"} <= set(task["combination"])
    backgrounds = {}
    if bg_case:
        for ref in task["references"]:
            if ref["semantic"] not in ("scene", "motion"):
                continue
            images = []
            for image in frames(ref)[0]:
                detections = model.detect(image, ["person", "animal", "vehicle"])
                background, _ = model.background(image, detections)
                if background is None and not detections and ref["semantic"] == "scene":
                    background = image
                if background is not None:
                    images.append(background)
            backgrounds[ref["handle"]] = model.embeddings(images)
    samples, bg_samples = defaultdict(list), defaultdict(list)
    for image in frames(candidate)[0]:
        detections = model.detect(image, labels)
        observed = model.embeddings([image.crop(tuple(d["box"])) for d in detections])
        for encoder in ("clip", "dino"):
            usable = [ref for ref in refs if encoder in targets[ref["handle"]]]
            matrix = np.full((len(usable), len(detections)), -2.0)
            for i, ref in enumerate(usable):
                if not detections:
                    continue
                values = (targets[ref["handle"]][encoder] @ observed[encoder].T).mean(0)
                for j, det in enumerate(detections):
                    if match_allowed(det["label"], categories(ref)):
                        matrix[i, j] = values[j]
            if matrix.size:
                ii, jj = linear_sum_assignment(-matrix)
                for i, j in zip(ii, jj):
                    if matrix[i, j] > -2:
                        samples[(usable[i]["handle"], encoder)].append(float(matrix[i, j]))
        if bg_case:
            background, _ = model.background(image, model.detect(image, ["person", "animal", "vehicle"]))
            value = model.embeddings([background]) if background is not None else {}
            for handle, target in backgrounds.items():
                for encoder in ("clip", "dino"):
                    if encoder in value and encoder in target:
                        bg_samples[(handle, encoder)].append(
                            float((value[encoder] @ target[encoder].T).mean())
                        )
    per_reference = defaultdict(dict)
    for (handle, encoder), values in samples.items():
        per_reference[handle]["localized_" + encoder] = float(np.mean(values))
    for (handle, encoder), values in bg_samples.items():
        per_reference[handle]["background_" + encoder] = float(np.mean(values))
    scores = role_means(task, per_reference)
    for encoder in ("clip", "dino"):
        a, b = scores.get("scene_background_" + encoder), scores.get("motion_background_" + encoder)
        if a is not None and b is not None:
            scores["scene_minus_motion_background_" + encoder] = a - b
    return {"scores": scores, "per_reference": dict(per_reference)}
