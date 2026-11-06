"""Pointwise diagnostics reported in the ORAV paper."""

from __future__ import annotations

import math
from collections import defaultdict
from importlib import metadata
from pathlib import Path

from ..io import digest, load_manifest, read_json, sha256_file, write_json

FAMILIES = (
    "visual",
    "vbench",
    "dover",
    "aesthetic",
    "audiobox",
    "speech",
    "music",
    "nisqa",
    "face",
    "style",
    "motion",
    "localization",
)
ROLES = {
    "speech": {"speech"},
    "music": {"music"},
    "nisqa": {"speech"},
    "face": {"subject"},
    "style": {"style"},
    "motion": {"motion"},
    "localization": {"subject", "prop", "scene"},
}


ASSETS = {
    "visual": ("clip_b32", "dino_source", "dino_b16"),
    "vbench": ("clip_b32", "clip_l14", "dino_source", "dino_b16", "laion_aesthetic", "amt", "musiq"),
    "dover": ("dover_source", "dover"),
    "aesthetic": ("aesthetic_source", "aesthetic_head", "siglip"),
    "audiobox": ("audiobox",),
    "speech": ("ecapa", "silero_vad", "whisper", "dnsmos"),
    "music": ("clap", "muq"),
    "nisqa": ("nisqa_source", "nisqa"),
    "face": ("retinaface", "arcface_source", "arcface"),
    "style": ("aladin_source", "aladin"),
    "motion": ("cotracker_source", "cotracker", "grounding_dino"),
    "localization": ("clip_b32", "dino_source", "dino_b16", "grounding_dino", "sam2"),
}


def asset_identity(path):
    if path.is_file():
        return {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if not path.is_dir():
        raise FileNotFoundError(f"Missing diagnostic asset: {path}")
    files = {
        str(p.relative_to(path)): sha256_file(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
        and not any(x.startswith(".") or x == "__pycache__" for x in p.relative_to(path).parts)
        and p.suffix not in (".pyc", ".pyo")
    }
    if not files:
        raise ValueError(f"Empty asset directory: {path}")
    return {"sha256": digest(files), "files": files}


def run(args):
    from .measure import load_model, measure

    config = read_json(args.config)
    assets = {
        key: (args.config.resolve().parent / value).resolve() for key, value in config["assets"].items()
    }
    if not 0 <= args.shard < args.shards:
        raise ValueError("Require 0 <= shard < shards")
    manifest = load_manifest(args.manifest)
    families = list(FAMILIES) if args.metrics == ["all"] else args.metrics
    if set(families) - set(FAMILIES):
        raise ValueError("Unknown metrics: " + str(set(families) - set(FAMILIES)))
    source = {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")}
    versions = {}
    for package in (
        "numpy",
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "vbench",
        "pyiqa",
        "speechbrain",
        "librosa",
        "audiobox-aesthetics",
        "muq",
        "open_clip_torch",
        "retinaface",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    fingerprints = {}
    output = Path(args.output)
    for family in families:
        selected = [
            task
            for task in manifest["tasks"]
            if family not in ROLES or ROLES[family] & set(task["combination"])
        ]
        selected = sorted(selected, key=lambda task: task["identity"])[args.shard :: args.shards]
        pending = [
            (task, system, record)
            for task in selected
            for system, record in task["candidates"].items()
            if record["state"] == "delivered"
        ]
        if not pending:
            continue
        print(f"{family}: recording asset checksums", flush=True)
        for key in ASSETS[family]:
            if key not in fingerprints:
                fingerprints[key] = asset_identity(assets[key])
        provenance = {
            "family": family,
            "source": source,
            "packages": versions,
            "assets": {key: fingerprints[key] for key in ASSETS[family]},
        }
        configuration = digest(provenance)
        write_json(output / "provenance" / (configuration + ".json"), provenance)
        model = None
        for index, (task, system, candidate) in enumerate(pending):
            identity = digest([configuration, family, task["identity"], system])
            path = output / family / (identity + ".json")
            if path.exists():
                continue
            if model is None:
                model = load_model(family, args.device, assets)
            row = {
                "configuration": configuration,
                "family": family,
                "task_identity": task["identity"],
                "task_id": task["key"],
                "system": system,
                "candidate_sha256": candidate["sha256"],
            }
            try:
                result = measure(family, model, task, candidate)
                if any(
                    not isinstance(x, (int, float)) or not math.isfinite(x) for x in result["scores"].values()
                ):
                    raise ValueError("Non-finite diagnostic score")
                row.update(result, state="ok")
            except Exception as error:
                row.update(state="error", scores={}, error=f"{type(error).__name__}: {error}")
            write_json(path, row)
            if hasattr(model, "cache"):
                model.cache.clear()
            print(f"{family}: {index + 1}/{len(pending)} {row['state']}", flush=True)
        del model


def summarize(args):
    import numpy as np

    manifest = load_manifest(args.manifest)
    tasks = {task["identity"]: task for task in manifest["tasks"]}
    systems = manifest["systems"]
    values, seen, configurations = defaultdict(dict), set(), defaultdict(set)
    errors = []
    for path in sorted(Path(args.results).glob("*/*.json")):
        row = read_json(path)
        if row.get("task_identity") not in tasks:
            continue
        key, system = row["task_identity"], row["system"]
        if system not in systems or tasks[key]["candidates"][system].get("sha256") != row["candidate_sha256"]:
            raise ValueError("Diagnostic candidate identity differs from manifest")
        record_key = (key, system, row["family"])
        if record_key in seen:
            raise ValueError("Duplicate task/system/family; use one result configuration")
        seen.add(record_key)
        configurations[row["family"]].add(row["configuration"])
        if len(configurations[row["family"]]) > 1:
            raise ValueError("Mixed configurations in one diagnostic family")
        if row["state"] != "ok":
            errors.append({"task": tasks[key]["key"], "system": system, "family": row["family"]})
        for metric, value in row.get("scores", {}).items():
            if math.isfinite(value):
                values[metric][key, system] = value
    quality = [
        "vbench_" + x
        for x in (
            "subject_consistency",
            "background_consistency",
            "motion_smoothness",
            "aesthetic_quality",
            "imaging_quality",
        )
    ]
    standardization = {}
    if all(name in values for name in quality):
        for name in quality:
            vector = list(values[name].values())
            standardization[name] = {
                "mean": float(np.mean(vector)),
                "std": float(np.std(vector, ddof=1)),
                "n": len(vector),
            }
        common = set.intersection(*(set(values[name]) for name in quality))
        for key in common:
            if all(standardization[name]["std"] > 0 for name in quality):
                values["comp_z"][key] = float(
                    np.mean(
                        [
                            (values[name][key] - standardization[name]["mean"]) / standardization[name]["std"]
                            for name in quality
                        ]
                    )
                )
    summary = {}
    for metric, entries in sorted(values.items()):
        shared = sorted(key for key in tasks if all((key, system) in entries for system in systems))
        summary[metric] = {
            "shared_tasks_n": len(shared),
            "means": {
                system: float(np.mean([entries[key, system] for key in shared])) if shared else None
                for system in systems
            },
            "task_ids": [tasks[key]["key"] for key in shared],
        }
    write_json(
        args.output,
        {"systems": systems, "metrics": summary, "corpus_standardization": standardization, "errors": errors},
    )
