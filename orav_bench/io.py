"""Manifest loading, content identities, and atomic result files."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import tempfile
from pathlib import Path

from PIL import Image


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def media_record(path, root, expected=None):
    path = (root / path).resolve()
    if not path.is_file():
        raise ValueError(f"Media file does not exist: {path}")
    sha = sha256_file(path)
    if expected and expected != sha:
        raise ValueError(f"Media SHA-256 mismatch: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    try:
        with Image.open(path) as image:
            mime = Image.MIME[image.format]
    except (OSError, KeyError):
        pass
    modality = mime.split("/")[0]
    if modality not in ("image", "video", "audio"):
        raise ValueError(f"Unsupported media type: {path}")
    return {"path": str(path), "sha256": sha, "mime_type": mime, "modality": modality}


def load_manifest(path):
    """Resolve paths relative to the manifest; labels are never evaluation inputs."""
    path = Path(path).resolve()
    value = read_json(path)
    if value.get("version") != 1 or not isinstance(value.get("tasks"), list):
        raise ValueError("Expected a version: 1 manifest with a tasks array")
    systems = value.get("systems")
    if not isinstance(systems, list) or len(systems) < 2 or len(set(systems)) != len(systems):
        raise ValueError("Declare at least two distinct system names")
    if any(not isinstance(x, str) or not x.strip() for x in systems):
        raise ValueError("System names must be nonempty strings")
    tasks, seen = [], set()
    roles = {"subject", "prop", "scene", "motion", "camera_movement", "vfx", "style", "speech", "music"}
    for raw in value["tasks"]:
        key = raw["id"]
        if not isinstance(key, str) or not key.strip() or key in seen:
            raise ValueError("Task IDs must be unique nonempty strings")
        seen.add(key)
        if not isinstance(raw["instruction"], str) or not raw["instruction"].strip():
            raise ValueError(f"Missing generation instruction: {key}")
        refs, handles = [], set()
        for ref in raw["references"]:
            handle, role = ref["handle"], ref["role"]
            if not isinstance(handle, str) or not handle or handle in handles or role not in roles:
                raise ValueError(f"Invalid reference handle or role: {key}")
            handles.add(handle)
            record = media_record(ref["path"], path.parent, ref.get("sha256"))
            if (role in ("speech", "music")) != (record["modality"] == "audio"):
                raise ValueError("Auditory references must be audio; visual references must be images/videos")
            refs.append({**record, "handle": handle, "semantic": role})
        candidates = {}
        if set(raw["candidates"]) - set(systems):
            raise ValueError(f"Undeclared system in task {key}")
        for system in systems:
            entry = raw["candidates"].get(system)
            if entry is None:
                candidates[system] = {"state": "not_delivered"}
                continue
            entry = {"path": entry} if isinstance(entry, str) else entry
            record = media_record(entry["path"], path.parent, entry.get("sha256"))
            if record["modality"] != "video":
                raise ValueError(f"Candidate must be a video: {key}/{system}")
            candidates[system] = {**record, "state": "delivered"}
        task = {
            "key": key,
            "instruction_en": raw["instruction"],
            "duration_s": raw.get("duration_s"),
            "references": refs,
            "candidates": candidates,
            "combination": sorted({ref["semantic"] for ref in refs}),
        }
        # Optional diagnostic metadata never enters a judge or listener request.
        task["diagnostics"] = raw.get("diagnostics", {})
        identity = {
            **task,
            "references": [{k: v for k, v in r.items() if k != "path"} for r in refs],
            "candidates": {s: {k: v for k, v in c.items() if k != "path"} for s, c in candidates.items()},
        }
        task["identity"] = digest(identity)
        tasks.append(task)
    if not tasks:
        raise ValueError("Manifest contains no tasks")
    return {"systems": systems, "tasks": tasks}
