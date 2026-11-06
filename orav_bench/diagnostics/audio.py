"""Voice affinity, speech errors, waveform reuse, and audio embeddings."""

import math
import re
import subprocess

import numpy as np
import torch
from opencc import OpenCC
from scipy.signal import correlate, correlation_lags, resample_poly

SIMPLIFY = OpenCC("t2s")


def decode(r):
    p = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            r["path"],
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        timeout=90,
    )
    if p.returncode:
        raise ValueError(p.stderr.decode(errors="replace")[-1000:])
    x = np.frombuffer(p.stdout, dtype=np.float32).copy()
    if not len(x):
        raise ValueError("no decoded audio samples")
    return x


def waveform(a, b):
    # Vectorized normalized correlation over valid temporal offsets.
    a = a[: len(a) // 4 * 4].reshape(-1, 4).mean(1).astype(np.float64)
    b = b[: len(b) // 4 * 4].reshape(-1, 4).mean(1).astype(np.float64)
    required = max(12000, math.ceil(min(len(a), len(b)) / 2))
    if min(len(a), len(b)) < required:
        return None
    lags = correlation_lags(len(b), len(a))
    keep = (lags >= -len(a) + required) & (lags <= len(b) - required)
    lags = lags[keep]
    dot = correlate(b, a, method="fft")[keep]
    x0 = np.maximum(0, -lags)
    y0 = np.maximum(0, lags)
    n = np.minimum(len(a) - x0, len(b) - y0)
    ea = np.r_[0, np.cumsum(a * a)]
    eb = np.r_[0, np.cumsum(b * b)]
    den = np.sqrt(np.maximum(0, (ea[x0 + n] - ea[x0]) * (eb[y0 + n] - eb[y0])))
    valid = den > 1e-12
    if not valid.any():
        return None
    scores = np.where(valid, dot / np.maximum(den, 1e-12), 0)
    i = int(np.argmax(np.abs(scores)))
    return {
        "abs_cosine": float(min(1, abs(scores[i]))),
        "signed_cosine": float(np.clip(scores[i], -1, 1)),
        "lag_s": float(lags[i] / 4000),
        "overlap_s": float(n[i] / 4000),
        "minimum_overlap_s": required / 4000,
    }


def tokens(s, characters=False):
    s = s.lower().replace("’", "'").replace("—", " ").replace("–", " ")
    if characters:
        s = SIMPLIFY.convert(s)
    return list("".join(re.findall(r"\w", s))) if characters else re.findall(r"\w+(?:'\w+)?", s)


def error_rate(target, actual):
    char = bool(re.search(r"[\u3400-\u9fff]", target))
    a, b = tokens(target, char), tokens(actual, char)
    if not a:
        return None
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = row
    return {
        "metric": "CER" if char else "WER",
        "edits": prev[-1],
        "reference_units": len(a),
        "value": prev[-1] / len(a),
    }


class Audio:
    def __init__(self, device, assets, role="speech"):
        self.device = device
        self.assets = assets
        self.cache = {}
        if role == "speech":
            from speechbrain.lobes.features import Fbank
            from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
            from speechbrain.processing.features import InputNormalization

            self.fbank = Fbank(n_mels=80).to(self.device)
            self.norm = InputNormalization(norm_type="sentence", std_norm=False).to(self.device)
            self.ecapa = (
                ECAPA_TDNN(
                    input_size=80,
                    channels=[1024, 1024, 1024, 1024, 3072],
                    kernel_sizes=[5, 3, 3, 3, 1],
                    dilations=[1, 2, 3, 4, 1],
                    attention_channels=128,
                    lin_neurons=192,
                    se_channels=128,
                )
                .to(self.device)
                .eval()
            )
            self.ecapa.load_state_dict(
                torch.load(assets["ecapa"], map_location=self.device, weights_only=True)
            )
            self.vad = torch.jit.load(str(assets["silero_vad"]), map_location="cpu").eval()
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            self.processor = WhisperProcessor.from_pretrained(assets["whisper"], local_files_only=True)
            self.asr = (
                WhisperForConditionalGeneration.from_pretrained(
                    assets["whisper"], local_files_only=True, torch_dtype=torch.float32
                )
                .to(self.device)
                .eval()
            )
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            self.mos = ort.InferenceSession(
                str(assets["dnsmos"]), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        else:
            from transformers import ClapFeatureExtractor, ClapModel

            self.clap = (
                ClapModel.from_pretrained(assets["clap"], local_files_only=True).to(self.device).eval()
            )
            self.clap_processor = ClapFeatureExtractor.from_pretrained(assets["clap"], local_files_only=True)

    @torch.inference_mode()
    def speech_segments(self, x):
        self.vad.reset_states()
        probs = []
        for start in range(0, len(x), 512):
            chunk = np.pad(x[start : start + 512], (0, max(0, start + 512 - len(x))))
            probs.append(float(self.vad(torch.from_numpy(chunk), 16000).item()))
        active = np.array(probs) >= 0.5
        spans = []
        start = None
        for i, v in enumerate(np.r_[active, False]):
            if v and start is None:
                start = i
            if not v and start is not None:
                if (i - start) * 512 >= 0.25 * 16000:
                    spans.append([max(0, start * 512 - 1600), min(len(x), i * 512 + 1600)])
                start = None
        merged = []
        for a, b in spans:
            if merged and a - merged[-1][1] <= 0.2 * 16000:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        return merged, probs

    @torch.inference_mode()
    def voice(self, x):
        a = torch.from_numpy(x.copy()).unsqueeze(0).to(self.device)
        f = self.norm(self.fbank(a), torch.ones(1, device=self.device))
        v = self.ecapa(f).flatten()
        return torch.nn.functional.normalize(v, dim=0).cpu().numpy()

    def dnsmos(self, x):
        size = int(9.01 * 16000)
        original = len(x)
        if original < 16000:
            return None
        y = x
        while len(y) < size:
            y = np.r_[y, y]
        vals = []
        for i in range(max(1, int(len(y) / 16000) - 9 + 1)):
            a = y[i * 16000 : i * 16000 + size]
            if len(a) < size:
                continue
            raw = np.asarray(self.mos.run(None, {"input_1": a[None].astype(np.float32)})[0]).reshape(-1)
            sig, bak, ovr = raw[:3]
            vals.append(
                [
                    float(np.polyval([-0.08397278, 1.22083953, 0.0052439], sig)),
                    float(np.polyval([-0.13166888, 1.60915514, -0.39604546], bak)),
                    float(np.polyval([-0.06766283, 1.11546468, 0.04602535], ovr)),
                ]
            )
        return {
            "SIG": float(np.mean(vals, axis=0)[0]),
            "BAK": float(np.mean(vals, axis=0)[1]),
            "OVRL": float(np.mean(vals, axis=0)[2]),
            "input_s": original / 16000,
            "repeated_to_minimum_window": original < size,
        }

    @torch.inference_mode()
    def transcribe(self, x):
        chunks = []
        for start in range(0, len(x), 30 * 16000):
            inputs = self.processor(
                x[start : start + 30 * 16000],
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            out = self.asr.generate(
                inputs.input_features.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                do_sample=False,
                num_beams=1,
                max_new_tokens=160,
                task="transcribe",
                return_timestamps=False,
            )
            chunks.append(self.processor.batch_decode(out, skip_special_tokens=True)[0].strip())
        return " ".join(chunks)

    @torch.inference_mode()
    def music(self, x):
        ee = []
        for start in range(0, len(x), 480000):
            a = x[start : start + 480000]
            inputs = self.clap_processor(a, sampling_rate=48000, return_tensors="pt")
            e = self.clap.get_audio_features(**{k: v.to(self.device) for k, v in inputs.items()})
            ee.append(e.cpu().numpy()[0])
        return np.asarray(ee)

    def media(self, r, role):
        k = (r["sha256"], role)
        if k in self.cache:
            return self.cache[k]
        x48 = decode(r)
        x = resample_poly(x48, 1, 3).astype(np.float32)
        rms = float(np.sqrt(np.mean(x.astype(float) ** 2)))
        result = {
            "sha256": r["sha256"],
            "duration_s": len(x) / 16000,
            "rms": rms,
            "clipping_fraction": float(np.mean(np.abs(x48) >= 0.999)),
            "state": "signal" if rms > 1e-5 else "silent",
        }
        value = {"result": result, "waveform": x, "voice": None, "music": None}
        if result["state"] == "signal":
            if role == "speech":
                spans, probs = self.speech_segments(x)
                result["vad_spans_s"] = [[a / 16000, b / 16000] for a, b in spans]
                result["speech_s"] = sum(b - a for a, b in spans) / 16000
                result["transcript"] = self.transcribe(x)
                if result["speech_s"] >= 1:
                    selected = np.concatenate([x[a:b] for a, b in spans])
                    value["voice"] = self.voice(selected)
                    result["voice_embedding"] = value["voice"].tolist()
                    a, b = max(spans, key=lambda ab: ab[1] - ab[0])
                    result["dnsmos_longest_speech_segment"] = self.dnsmos(x[a:b])
                else:
                    result["unavailable_reason"] = "less_than_1s_detected_speech"
            else:
                value["music"] = self.music(x48)
                result["clap_embeddings"] = value["music"].tolist()
        self.cache[k] = value
        return value
