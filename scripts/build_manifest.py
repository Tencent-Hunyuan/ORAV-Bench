"""Join task definitions, prompt files and selected model outputs without copying media."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from orav_bench.io import digest, read_json, sha256_file, write_json


def benchmark_fingerprint(tasks_path):
    tasks_path = Path(tasks_path)
    return digest({
        "tasks": sha256_file(tasks_path),
        "prompts": {
            task_id: sha256_file(tasks_path.parent / "tasks" / task_id / "prompt.en.txt")
            for task_id in read_json(tasks_path)
        },
    })


def build_manifest(tasks_path, generation_paths, destination):
    tasks_path = Path(tasks_path).resolve()
    generation_paths = [Path(path).resolve() for path in generation_paths]
    destination = Path(destination).resolve()
    bindings = [path.parent.parent / "benchmark.json" for path in generation_paths]
    if destination in [tasks_path, *generation_paths, *bindings]:
        raise ValueError("The evaluation manifest must not overwrite source metadata")
    definitions = read_json(tasks_path)
    if not definitions:
        raise ValueError("Expected nonempty benchmark tasks")
    fingerprint = benchmark_fingerprint(tasks_path)
    indexed = {}
    for task_id, definition in definitions.items():
        folder = tasks_path.parent / "tasks" / task_id
        indexed[task_id] = {
            "id": task_id,
            "instruction": (folder / "prompt.en.txt").read_text(encoding="utf-8"),
            "duration_s": definition["duration_s"],
            "references": [
                {"handle": ref["handle"], "role": ref["role"], "sha256": ref["sha256"],
                 "path": os.path.relpath(folder / ref["file"], destination.parent)}
                for ref in definition["references"]
            ],
            "candidates": {},
        }
    systems, sources = [], []
    binding = None
    for path, binding_path in zip(generation_paths, bindings):
        current = read_json(binding_path)
        if not current.get("repo_id") or not re.fullmatch(r"[0-9a-f]{40}", current.get("revision", "")):
            raise ValueError("Artifacts must pin a benchmark repository and revision")
        if current.get("input_sha256") != fingerprint:
            raise ValueError("Artifacts do not match the selected tasks and English prompts")
        if binding is not None and current != binding:
            raise ValueError("Selected outputs refer to different benchmark versions")
        binding = current
        system = path.parent.name
        if not system.strip() or system in systems:
            raise ValueError("Select distinct system folders")
        systems.append(system)
        manifest = read_json(path)
        completed, failed = manifest["completed"], manifest["failed"]
        declared = [*completed, *failed]
        if len(set(declared)) != len(declared) or set(declared) - set(indexed):
            raise ValueError("Unknown or duplicate output task")
        delivered = set(completed)
        for task_id, task in indexed.items():
            task["candidates"][system] = (
                os.path.relpath(path.parent / "videos" / (task_id + ".mp4"), destination.parent)
                if task_id in delivered else None
            )
        sources.append({"path": str(path), "sha256": sha256_file(path)})
    if len(systems) < 2:
        raise ValueError("Select at least two systems")
    value = {
        "version": 1, "benchmark": binding, "generation_manifests": sources,
        "systems": systems, "tasks": list(indexed.values()),
    }
    write_json(destination, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="Benchmark tasks.json")
    parser.add_argument("--generations", type=Path, nargs="+", required=True, help="Selected model manifests")
    parser.add_argument("--output", type=Path, required=True, help="New evaluation manifest")
    args = parser.parse_args()
    value = build_manifest(args.tasks, args.generations, args.output)
    print(f"Saved {len(value['tasks'])} tasks and {len(value['systems'])} systems to {args.output}")


if __name__ == "__main__":
    main()
