"""Timestamped visual panels and source-audio evidence."""

from __future__ import annotations

import fcntl
import json
import math
import mimetypes
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

from .io import digest, read_json, sha256_file, write_json
from .protocol import PROMPTS, compact

MIN_WINDOW_S = 0.02
MAX_BINS = 256
FLOOR_DBFS = -100.0


@dataclass(frozen=True)
class Config:
    max_panels: int = 6
    base_frames_per_panel: int = 8
    max_canvas_side: int = 2048
    label_height: int = 24
    jpeg_quality: int = 90

    def validate(self):
        if any(
            type(x) is not int or x < 1
            for x in (self.max_panels, self.base_frames_per_panel, self.max_canvas_side, self.label_height)
        ):
            raise ValueError("positive integer packing bounds required")
        if self.max_canvas_side <= self.label_height or not 1 <= self.jpeg_quality <= 95:
            raise ValueError("invalid canvas/encoding bound")


def validate_frames(frames):
    if not frames:
        raise ValueError("empty frame sequence")
    times = [f["time_s"] for f in frames]
    indices = [f["index"] for f in frames]
    if (
        any(not math.isfinite(t) for t in times)
        or any(a >= b for a, b in zip(times, times[1:]))
        or any(a >= b for a, b in zip(indices, indices[1:]))
    ):
        raise ValueError("strict frame/time chronology required; do not reorder silently")


def partition(frames, cfg=Config()):
    cfg.validate()
    validate_frames(frames)
    size = max(cfg.base_frames_per_panel, math.ceil(len(frames) / cfg.max_panels))
    chunks = [frames[i : i + size] for i in range(0, len(frames), size)]
    if len(chunks) > cfg.max_panels or [f for c in chunks for f in c] != frames:
        raise ValueError("panel cap or exact frame preservation failed")
    return size, chunks


def layout(count, width, height, cfg=Config()):
    cfg.validate()
    if min(count, width, height) < 1:
        raise ValueError("positive image dimensions and frame count required")
    choices = []
    for columns in range(1, count + 1):
        rows = math.ceil(count / columns)
        room = cfg.max_canvas_side - rows * cfg.label_height
        if room < rows:
            continue
        scale = min(1.0, cfg.max_canvas_side / (columns * width), room / (rows * height))
        w, h = math.floor(width * scale), math.floor(height * scale)
        if min(w, h) < 1:
            continue
        canvas = [columns * w, rows * (h + cfg.label_height)]
        score = (-scale, max(canvas), rows * columns - count, abs(canvas[0] - canvas[1]), columns)
        choices.append(
            (
                score,
                {
                    "columns": columns,
                    "rows": rows,
                    "tile_width": w,
                    "tile_height": h,
                    "source_width": width,
                    "source_height": height,
                    "canvas_width": canvas[0],
                    "canvas_height": canvas[1],
                    "nominal_linear_scale": scale,
                    "resized": (w, h) != (width, height),
                },
            )
        )
    if not choices:
        raise ValueError("no positive-size layout fits canvas; no frames dropped")
    return min(choices, key=lambda x: x[0])[1]


@dataclass(frozen=True)
class Typography:
    font_size: int = 26
    min_font_size: int = 10
    horizontal_padding: int = 4
    vertical_padding: int = 2

    def validate(self):
        values = asdict(self)
        if (
            any(type(v) is not int for v in values.values())
            or not 1 <= self.min_font_size <= self.font_size
            or min(self.horizontal_padding, self.vertical_padding) < 0
        ):
            raise ValueError("invalid label font size or padding")


def label_image(text, width, height, typography=Typography()):
    """Fit the complete text; fail explicitly if even the minimum cannot fit."""
    typography.validate()
    if min(width, height) < 1:
        raise ValueError("positive label dimensions required")
    image = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    pad_x, pad_y = typography.horizontal_padding, typography.vertical_padding
    for size in range(typography.font_size, typography.min_font_size - 1, -1):
        font = ImageFont.load_default(size=size)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        w, h = right - left, bottom - top
        if w <= width - 2 * pad_x and h <= height - 2 * pad_y:
            x, y = pad_x - left, (height - h) // 2 - top
            draw.text((x, y), text, font=font, fill="white")
            bbox = list(draw.textbbox((x, y), text, font=font))
            return image, {
                "font_size": size,
                "glyph_bbox": bbox,
                "glyph_height": h,
                "font_reduced_to_fit": size < typography.font_size,
                "label_clipped": False,
            }
    raise ValueError(f"complete timestamp cannot fit {width}x{height} label; no truncation")


def render(frames, destination, cfg=Config(), typography=Typography()):
    size, chunks = partition(frames, cfg)
    typography.validate()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("refuse to overwrite a prepared panel directory")
    for frame in frames:
        if sha256_file(Path(frame["path"])) != frame["sha256"]:
            raise ValueError("source frame SHA changed")
    panels = []
    for panel_index, chunk in enumerate(chunks):
        with Image.open(chunk[0]["path"]) as first:
            source_size = first.size
        geometry = layout(len(chunk), *source_size, cfg)
        canvas = Image.new("RGB", (geometry["canvas_width"], geometry["canvas_height"]), (20, 20, 20))
        roster = []
        for position, frame in enumerate(chunk):
            w, h = geometry["tile_width"], geometry["tile_height"]
            x = position % geometry["columns"] * w
            y = position // geometry["columns"] * (h + cfg.label_height)
            with Image.open(frame["path"]) as original:
                if original.size != source_size:
                    raise ValueError("frame geometry differs inside one clip")
                original = original.convert("RGB")
                if original.size != (w, h):
                    original = original.resize((w, h), Image.Resampling.LANCZOS)
                canvas.paste(original, (x, y + cfg.label_height))
            text = f"frame {frame['index']}  t={frame['time_s']:.6f}s"
            label, metrics = label_image(text, w, cfg.label_height, typography)
            canvas.paste(label, (x, y))
            roster.append(
                {
                    **{k: frame[k] for k in ("index", "time_s", "sha256")},
                    "tile": {"x": x, "y": y + cfg.label_height, "width": w, "height": h},
                    "label": text,
                    **metrics,
                }
            )
        path = destination / f"panel-{panel_index:03d}.jpg"
        canvas.save(path, "JPEG", quality=cfg.jpeg_quality)
        panels.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "geometry": geometry,
                "frames": roster,
            }
        )
    actual = [(f["index"], f["time_s"], f["sha256"]) for p in panels for f in p["frames"]]
    if actual != [(f["index"], f["time_s"], f["sha256"]) for f in frames]:
        raise ValueError("output lost, repeated or reordered source frames")
    return {
        "config": asdict(cfg),
        "typography": asdict(typography),
        "pillow_version": PIL.__version__,
        "selected_frame_count": len(frames),
        "frames_per_panel_target": size,
        "panel_count": len(panels),
        "panels": panels,
        "raw_panel_bytes": sum(p["bytes"] for p in panels),
        "base64_media_characters": sum(4 * math.ceil(p["bytes"] / 3) for p in panels),
        "all_selected_frames_preserved": True,
        "chronology_preserved": True,
        "any_local_tile_downscaled": any(p["geometry"]["resized"] for p in panels),
        "any_timestamp_label_clipped": False,
    }


def summarize(samples, rate, start_s=0.0):
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not values.size or not np.isfinite(values).all() or rate <= 0:
        raise ValueError("invalid decoded samples")
    width = max(math.ceil(rate * MIN_WINDOW_S), math.ceil(len(values) / MAX_BINS))
    db = []
    for offset in range(0, len(values), width):
        square = float(np.mean(values[offset : offset + width] ** 2))
        db.append(round(max(FLOOR_DBFS, 10 * math.log10(square)) if square > 0 else FLOOR_DBFS, 2))
    return {
        "start_s": round(start_s, 6),
        "end_s": round(start_s + len(values) / rate, 6),
        "window_s": width / rate,
        "window_samples": width,
        "sample_rate": rate,
        "channels": values.shape[1],
        "decoded_frames": len(values),
        "rms_dbfs": db,
        "floor_dbfs": FLOOR_DBFS,
        "last_bin_samples": len(values) - width * (len(db) - 1),
    }


def audio_frames(log):
    rows = []
    for line in log.splitlines():
        if "ashowinfo" not in line or "nb_samples:" not in line:
            continue
        fields = {
            key: re.search(r"\b" + key + r":\s*([^ ]+)", line)
            for key in ("pts_time", "channels", "rate", "nb_samples")
        }
        if not all(fields.values()):
            raise ValueError("decoded frame clock metadata missing")
        rows.append(
            {
                "time": float(fields["pts_time"][1]),
                "channels": int(fields["channels"][1]),
                "rate": int(fields["rate"][1]),
                "samples": int(fields["nb_samples"][1]),
            }
        )
    if not rows:
        raise ValueError("no decoded frame clock")
    rate, channels = rows[0]["rate"], rows[0]["channels"]
    for before, after in zip(rows, rows[1:]):
        if (after["rate"], after["channels"]) != (rate, channels):
            raise ValueError("audio format changes within clip")
        expected = before["time"] + before["samples"] / rate
        if abs(after["time"] - expected) > max(2 / rate, 0.00015):
            raise ValueError("discontinuous audio clock: no invented contiguous timeline")
    return rows


def command(args, timeout=300):
    result = subprocess.run([str(x) for x in args], capture_output=True, timeout=timeout)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace")[-1200:])
    return result


def probe(path):
    value = json.loads(
        command(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path]).stdout
    )
    videos = [x for x in value["streams"] if x["codec_type"] == "video"]
    audio = [x for x in value["streams"] if x["codec_type"] == "audio"]
    duration = float(value.get("format", {}).get("duration") or (videos[0].get("duration") if videos else 0))
    return {"duration_s": duration, "has_audio_stream": bool(audio), "video": videos, "audio": audio}


def attachment(path, label, mime):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path), "label": label, "mime_type": mime}


def text_attachment(label, text):
    return {"label": label, "mime_type": "text/plain", "text": text}


def visual_proxy(path, destination, height, budget):
    """Full-duration, fixed-profile proxy; original audio is handled separately."""
    info = probe(path)
    if info["duration_s"] <= 0 or budget < 32000:
        raise ValueError("Invalid video duration or proxy budget")
    base = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-n",
        "-i",
        path,
        "-vf",
        f"scale=-2:min({height}\\,ih),format=yuv420p",
        "-c:v",
        "libx264",
        "-profile:v",
        "main",
        "-preset",
        "medium",
        "-bf",
        "0",
    ]
    for index, factor in enumerate((None, 0.80, 0.60, 0.40)):
        output = destination / f"proxy-{index}.mp4"
        if not output.exists():
            if factor is None:
                options = [
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "22",
                    "-g",
                    "48",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                ]
            else:
                rate = int(budget * 8 * factor / info["duration_s"]) - (
                    48000 if info["has_audio_stream"] else 0
                )
                if rate < 16000:
                    raise ValueError("Full-duration video cannot fit the proxy budget")
                options = [
                    "-b:v",
                    str(rate),
                    "-maxrate",
                    str(rate),
                    "-bufsize",
                    str(rate * 2),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "48k",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                ]
            command(base + options + ["-movflags", "+faststart", output])
        if output.stat().st_size <= budget:
            return output
    raise ValueError("Full-duration video exceeds its declared byte budget")


def prepare_media(record, output, *, reference=False, budget=4000000):
    """Prepare one source independent of the opposing candidate."""
    source = Path(record["path"])
    if sha256_file(source) != record["sha256"]:
        raise ValueError("Source media changed")
    home = Path(output) / "media" / digest([record["sha256"], reference, budget, "panels-v1"])
    home.mkdir(parents=True, exist_ok=True)
    with (home / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare_media(record, source, home, reference, budget)


def _prepare_media(record, source, home, reference, budget):
    receipt = home / "media.json"
    if receipt.exists():
        value = read_json(receipt)
        for item in value["images"] + ([value["audio"]] if value["audio"] else []):
            if sha256_file(item["path"]) != item["sha256"]:
                raise ValueError("Prepared media changed")
        return value
    # A missing receipt means an interrupted preparation; regenerate derived files.
    for child in home.iterdir():
        if child.name != "lock":
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    images, sound, energy = [], None, {"status": "no_audio_stream"}
    if record["modality"] == "image":
        path = home / "image.jpg"
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((1024, 1024))
            image.save(path, "JPEG", quality=90)
        images.append(attachment(path, "", "image/jpeg"))
        info = {"duration_s": None, "has_audio_stream": False}
    else:
        info = probe(source)
        if record["modality"] == "video":
            proxy = visual_proxy(source, home, 480 if reference else 576, budget)
            directory = home / "frames"
            directory.mkdir(exist_ok=True)
            # All nominal 8-fps samples are retained; showinfo supplies decoded timestamps.
            result = command(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-v",
                    "info",
                    "-threads",
                    "1",
                    "-y",
                    "-i",
                    proxy,
                    "-an",
                    "-vf",
                    "select='gte(t,selected_n/8)',scale=512:-2,showinfo",
                    "-fps_mode",
                    "passthrough",
                    "-q:v",
                    "3",
                    "-threads",
                    "1",
                    directory / "frame-%05d.jpg",
                ]
            )
            times = [
                float(x)
                for x in re.findall(
                    r"Parsed_showinfo[^\n]*\bn:\s*\d+[^\n]*pts_time:([\d.eE+\-]+)",
                    result.stderr.decode(errors="replace"),
                )
            ]
            paths = sorted(directory.glob("frame-*.jpg"))
            if not times or len(paths) != len(times):
                raise ValueError("Frame/timestamp count differs")
            rows = [
                {"path": str(path), "sha256": sha256_file(path), "index": i, "time_s": t}
                for i, (path, t) in enumerate(zip(paths, times))
            ]
            packing = render(rows, home / "panels")
            for panel in packing["panels"]:
                roster = ", ".join(f"{f['index']}@{f['time_s']:.6f}s" for f in panel["frames"])
                images.append({**attachment(panel["path"], "", "image/jpeg"), "frames": roster})
        if info["has_audio_stream"]:
            wav = home / "audio.wav"
            if not wav.exists():
                command(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-n",
                        "-i",
                        source,
                        "-map",
                        "0:a:0",
                        "-vn",
                        "-c:a",
                        "pcm_f32le",
                        wav,
                    ]
                )
            if reference and record["modality"] == "audio":
                if source.stat().st_size > budget:
                    raise ValueError("Reference audio exceeds its byte budget")
                sound = attachment(source, "", mimetypes.guess_type(source)[0] or "audio/wav")
            else:
                sound = attachment(wav, "", "audio/wav")
            decoded = command(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-v",
                    "info",
                    "-copyts",
                    "-i",
                    source,
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-af",
                    "aformat=sample_fmts=flt,ashowinfo",
                    "-c:a",
                    "pcm_f32le",
                    "-f",
                    "f32le",
                    "pipe:1",
                ]
            )
            clocks = audio_frames(decoded.stderr.decode(errors="replace"))
            rate, channels = clocks[0]["rate"], clocks[0]["channels"]
            samples = np.frombuffer(decoded.stdout, dtype="<f4")
            if samples.size != sum(r["samples"] for r in clocks) * channels:
                raise ValueError("Decoded samples and audio timestamps disagree")
            origin = clocks[0]["time"]
            if record["modality"] == "video":
                video = command(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-hide_banner",
                        "-v",
                        "info",
                        "-copyts",
                        "-i",
                        source,
                        "-map",
                        "0:v:0",
                        "-an",
                        "-vf",
                        "showinfo",
                        "-frames:v",
                        "1",
                        "-f",
                        "null",
                        "-",
                    ]
                )
                match = re.search(
                    r"Parsed_showinfo[^\n]*pts_time:([\d.eE+\-]+)", video.stderr.decode(errors="replace")
                )
                if not match:
                    raise ValueError("Missing video timeline origin")
                origin = float(match[1])
            energy = {
                "status": "measured",
                "timeline": summarize(samples.reshape(-1, channels), rate, clocks[0]["time"] - origin),
            }
    value = {
        "source_sha256": record["sha256"],
        "images": images,
        "audio": sound,
        "streams": {k: info[k] for k in ("duration_s", "has_audio_stream")},
        "energy": energy,
    }
    write_json(receipt, value)
    return value


def prepare_task(task, output):
    refs = {}
    for ref in task["references"]:
        if ref["modality"] != "video":
            refs[ref["handle"]] = prepare_media(ref, output, reference=True, budget=6000000)
    used = sum(
        Path(x["path"]).stat().st_size
        for ref in task["references"]
        if ref["modality"] != "video"
        for x in (refs[ref["handle"]]["images"] or [refs[ref["handle"]]["audio"]])
        if x
    )
    videos = sum(ref["modality"] == "video" for ref in task["references"])
    for ref in task["references"]:
        if ref["modality"] == "video":
            refs[ref["handle"]] = prepare_media(
                ref, output, reference=True, budget=(6000000 - used) // videos
            )
    candidates = {
        name: prepare_media(c, output) for name, c in task["candidates"].items() if c["state"] == "delivered"
    }
    return {"references": refs, "candidates": candidates}


def judge_attachments(task, bundle, systems):
    result = []
    blocks = [
        (f"R{i}", f"REFERENCE {r['handle']} ({r['semantic']})", bundle["references"][r["handle"]])
        for i, r in enumerate(task["references"])
        if r["modality"] != "audio"
    ]
    blocks += [
        (slot, "CANDIDATE " + slot, bundle["candidates"][system]) for slot, system in zip("AB", systems)
    ]
    for mid, role, block in blocks:
        for i, image in enumerate(block["images"]):
            label = f"MEDIA {mid} | {role} | {mid}_P{i:03d}"
            if image.get("frames"):
                label += " chronological frames " + image["frames"]
            result.append({k: v for k, v in image.items() if k != "frames"} | {"label": label})
    result.append(
        text_attachment(
            "TIMESTAMPED PANEL GUIDE",
            "Image inputs only. No native video or sound is supplied to this\n"
            "extractor. Read each panel left-to-right, then top-to-bottom. Times are local to\n"
            "its MEDIA ID. Do not infer events between samples. Cite the supplied panel/image IDs.\n",
        )
    )
    result.append(
        text_attachment(
            "MEASURED CANDIDATE STREAMS",
            compact({s: bundle["candidates"][n]["streams"] for s, n in zip("AB", systems)}),
        )
    )
    result.append(
        text_attachment(
            "MEASURED CANDIDATE AUDIO ENERGY",
            PROMPTS["energy"]
            + "\n"
            + compact({s: bundle["candidates"][n]["energy"] for s, n in zip("AB", systems)}),
        )
    )
    return result


def listener_attachments(task, bundle, system):
    blocks = [
        (f"R{i}", f"REFERENCE {r['handle']} ({r['semantic']})", bundle["references"][r["handle"]])
        for i, r in enumerate(task["references"])
        if r["semantic"] in ("speech", "music")
    ]
    blocks.append(("M0", "CANDIDATE M0", bundle["candidates"][system]))
    result = [
        {**block["audio"], "label": f"MEDIA {mid} | {role}"} for mid, role, block in blocks if block["audio"]
    ]
    result.append(
        text_attachment(
            "MEASURED MEDIA STREAMS",
            compact({mid: {"role": role, **block["streams"]} for mid, role, block in blocks}),
        )
    )
    result.append(
        text_attachment(
            "MEASURED AUDIO ENERGY TIMELINES",
            PROMPTS["listener_energy"]
            + "\n"
            + compact({mid: block["energy"] for mid, _, block in blocks if block["audio"]}),
        )
    )
    return result
