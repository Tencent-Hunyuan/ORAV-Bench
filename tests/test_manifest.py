import runpy
from pathlib import Path

import pytest

from orav_bench.io import read_json, write_json

module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/build_manifest.py"))
build_manifest = module["build_manifest"]
benchmark_fingerprint = module["benchmark_fingerprint"]


@pytest.fixture
def inputs(tmp_path):
    tasks = tmp_path / "benchmark/tasks.json"
    definitions = {}
    for task in ("one", "two"):
        folder = tasks.parent / "tasks" / task
        folder.mkdir(parents=True)
        (folder / "prompt.en.txt").write_text("Follow @image1.\n")
        definitions[task] = {"duration_s": 5, "references": [
            {"handle": "@image1", "role": "subject", "file": "refs/person.jpg", "sha256": "f" * 64}
        ]}
    write_json(tasks, definitions)
    artifacts = tmp_path / "generations"
    write_json(artifacts / "benchmark.json", {
        "repo_id": "owner/tasks", "revision": "a" * 40, "input_sha256": benchmark_fingerprint(tasks),
    })
    paths = []
    for system in ("model-a", "model-b"):
        path = artifacts / system / "manifest.json"
        write_json(path, {"config": {}, "completed": ["one"], "failed": []})
        paths.append(path)
    return tasks, paths, tmp_path / "runs/example/manifest.json"


def test_join_reads_txt_and_preserves_sources_idempotently(inputs):
    tasks, paths, output = inputs
    first = read_json(paths[0])
    first["failed"].append("two")
    write_json(paths[0], first)
    before = {path: path.read_bytes() for path in [tasks, *paths]}
    value = build_manifest(tasks, paths, output)
    saved = output.read_bytes()
    build_manifest(tasks, paths, output)
    assert output.read_bytes() == saved
    assert {path: path.read_bytes() for path in before} == before
    assert value["systems"] == ["model-a", "model-b"]
    assert value["tasks"][0]["instruction"] == "Follow @image1.\n"
    assert value["tasks"][1]["candidates"] == {"model-a": None, "model-b": None}
    row = value["tasks"][0]
    assert (output.parent / row["references"][0]["path"]).resolve() == tasks.parent / "tasks/one/refs/person.jpg"
    assert (output.parent / row["candidates"]["model-a"]).resolve() == paths[0].parent / "videos/one.mp4"
    assert read_json(paths[0])["failed"] == ["two"]


@pytest.mark.parametrize("changed", ["tasks", "prompt"])
def test_rejects_changed_benchmark_inputs(inputs, changed):
    tasks, paths, output = inputs
    if changed == "tasks":
        definitions = read_json(tasks)
        definitions["one"]["duration_s"] = 10
        write_json(tasks, definitions)
    else:
        (tasks.parent / "tasks/one/prompt.en.txt").write_text("Different instruction.")
    with pytest.raises(ValueError, match="do not match"):
        build_manifest(tasks, paths, output)
    assert not output.exists()


def test_rejects_unpinned_or_mixed_versions(inputs):
    tasks, paths, output = inputs
    binding = paths[0].parent.parent / "benchmark.json"
    data = read_json(binding)
    data["revision"] = "main"
    write_json(binding, data)
    with pytest.raises(ValueError, match="must pin"):
        build_manifest(tasks, paths, output)
    data["revision"] = "a" * 40
    write_json(binding, data)
    other = output.parent / "other/model-b/manifest.json"
    write_json(other, read_json(paths[1]))
    write_json(other.parent.parent / "benchmark.json", {**data, "revision": "d" * 40})
    with pytest.raises(ValueError, match="different benchmark versions"):
        build_manifest(tasks, [paths[0], other], output)


def test_rejects_unknown_tasks_duplicate_models_and_source_overwrite(inputs):
    tasks, paths, output = inputs
    with pytest.raises(ValueError, match="distinct"):
        build_manifest(tasks, [paths[0], paths[0]], output)
    value = read_json(paths[0])
    value["failed"].append("unknown")
    write_json(paths[0], value)
    with pytest.raises(ValueError, match="Unknown"):
        build_manifest(tasks, paths, output)
    with pytest.raises(ValueError, match="must not overwrite"):
        build_manifest(tasks, paths, tasks)
