"""Shared decoding and measurement utilities."""

import json
import re
import subprocess

import numpy as np
from PIL import Image

from ..media import command


def probe(path):
    return json.loads(
        command(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path]).stdout
    )


def frames(media, n=8, size=640):
    path = media["path"]
    if media.get("modality") == "image":
        return [Image.open(path).convert("RGB")], [0.0]
    info = probe(path)
    s = next(x for x in info["streams"] if x["codec_type"] == "video")
    duration = float(s.get("duration") or info["format"]["duration"])
    w, h = s["width"], s["height"]
    ratio = size / max(w, h)
    w, h = round(w * ratio), round(h * ratio)
    filt = f"select='gte(t,(selected_n+0.5)*{duration}/{n})',showinfo,scale={w}:{h}"
    cmd = [
        "ffmpeg",
        "-v",
        "info",
        "-threads",
        "1",
        "-i",
        path,
        "-an",
        "-vf",
        filt,
        "-fps_mode",
        "passthrough",
        "-frames:v",
        str(n),
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    p = subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    a = np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)
    ts = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", p.stderr.decode(errors="replace"))][: len(a)]
    if len(a) != n or len(ts) != n:
        raise ValueError(f"expected {n} frames; decoded {len(a)} timestamps {len(ts)}")
    return [Image.fromarray(x) for x in a], ts


def audio(media, sr=16000, stereo=False):
    info = probe(media["path"])
    streams = [s for s in info["streams"] if s["codec_type"] == "audio"]
    if not streams:
        return None, {"state": "no_audio_stream"}
    channels = 2 if stereo else 1
    p = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            media["path"],
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sr),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
        timeout=90,
    )
    x = np.frombuffer(p.stdout, np.float32).copy().reshape(-1, channels).T
    if x.shape[1] == 0:
        return None, {"state": "empty_audio"}
    rms = float(np.sqrt(np.mean(x.astype(float) ** 2)))
    meta = {
        "state": "signal" if rms > 1e-5 else "silent",
        "duration_s": x.shape[1] / sr,
        "sample_rate": sr,
        "original_channels": streams[0].get("channels"),
        "rms": rms,
        "clipping_fraction": float(np.mean(np.abs(x) >= 0.999)),
    }
    return x, meta


def temporal(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(a):
        return {"mean": None, "valid_frames": 0, "total_frames": len(values)}
    early = [v for v in values[: max(1, len(values) // 3)] if v is not None]
    late = [v for v in values[-max(1, len(values) // 3) :] if v is not None]
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "minimum": float(a.min()),
        "p10": float(np.quantile(a, 0.1)),
        "late_minus_early": float(np.mean(late) - np.mean(early)) if early and late else None,
        "valid_frames": len(a),
        "total_frames": len(values),
    }
