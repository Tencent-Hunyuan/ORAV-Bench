import copy
import json
import shutil
import subprocess

import numpy as np
import pytest

from orav_bench import protocol, ranking
from orav_bench.client import Client
from orav_bench.io import digest, load_manifest, read_json, write_json


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is required")
def test_frame_decoding_keeps_selected_timestamps(tmp_path):
    from orav_bench.diagnostics.common import frames

    path = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=16:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ], check=True, timeout=30)
    images, timestamps = frames({"path": str(path), "modality": "video"}, n=4, size=96)
    assert len(images) == 4
    assert all(image.size == (96, 64) for image in images)
    assert timestamps == pytest.approx([0.125, 0.375, 0.625, 0.875], abs=1e-6)


def response(spec):
    if "enum" in spec:
        return spec["enum"][0]
    kind = spec["type"]
    if kind == "object":
        return {k: response(v) for k, v in spec["properties"].items()}
    if kind == "array":
        return []
    if kind == "boolean":
        return True
    if kind in ("number", "integer"):
        return 0
    return "Observed evidence."


def task(speech=False):
    refs = [{"handle": "@image1", "semantic": "subject", "modality": "image"}]
    if speech:
        refs.append({"handle": "@audio1", "semantic": "speech", "modality": "audio"})
    return {
        "key": "example",
        "identity": "example",
        "instruction_en": "Say hello while following @image1.",
        "duration_s": 4,
        "references": refs,
    }


def judgment(winner="A"):
    value = response(protocol.schema("judge", task()))
    value["overall_task"]["winner"] = winner
    return value


def test_fusion_keeps_tie_conflict_missing_and_unobservable_separate():
    systems = ["first", "second"]
    rows = [
        {"status": "ok", "systems": systems, "direction": "AB", "parsed": judgment("A")},
        {"status": "ok", "systems": systems[::-1], "direction": "BA", "parsed": judgment("B")},
    ]
    assert protocol.fuse(rows, systems)["winner"] == "first"
    rows[1]["parsed"]["overall_task"]["winner"] = "A"
    assert protocol.fuse(rows, systems)["status"] == "order_conflict"
    rows[1]["parsed"]["overall_task"]["evidence_sufficient"] = False
    assert protocol.fuse(rows, systems)["status"] == "not_observable"
    rows[1]["status"] = "missing"
    assert protocol.fuse(rows, systems)["status"] == "missing"
    for row in rows:
        row.update(status="ok", parsed=judgment("tie"))
        row["parsed"]["quality_tiebreak"]["winner"] = "tie"
    assert protocol.fuse(rows, systems)["winner"] == "tie"


def test_usability_precedes_overall_and_quality_only_breaks_task_ties():
    value = judgment("A")
    value["usability"]["A"]["usable"] = False
    assert protocol.decision(value, ["x", "y"])["winner"] == "y"
    value["usability"]["A"]["evidence_sufficient"] = False
    value["quality_tiebreak"]["winner"] = "B"
    assert protocol.decision(value, ["x", "y"])["winner"] == "x"
    value["overall_task"]["winner"] = "tie"
    assert protocol.decision(value, ["x", "y"])["winner"] == "y"


def test_speech_mapping_replay_and_winner_only_consistency():
    value = response(protocol.schema("voice"))
    value.update(winner="A", gap="material")
    reversed_value = copy.deepcopy(value)
    reversed_value.update(winner="B", gap="small")
    rows = [
        {"direction": "AB", "systems": ["x", "y"], "parsed": value},
        {"direction": "BA", "systems": ["y", "x"], "parsed": reversed_value},
    ]
    transcripts = {
        name: {
            "transcript": "hello",
            "speech_state": "intelligible",
            "evidence": "Heard hello.",
            "limitations": [],
        }
        for name in ("x", "y")
    }
    prompt, _, context = protocol.judge_input(task(True), ["x", "y"], "AB", transcripts, rows)
    assert context["kind"] == "judge_speech"
    assert "INDEPENDENT CANDIDATE SPEECH CONTENT" in prompt
    rows[0]["parsed"]["candidates"]["A"]["shared_and_different"] = (
        "The candidate replayed the reference recording."
    )
    prompt, _, _ = protocol.judge_input(task(True), ["x", "y"], "AB", transcripts, rows)
    assert protocol.PROMPTS["replay_note"] in prompt
    rows[1]["parsed"]["winner"] = "A"
    _, _, context = protocol.judge_input(task(True), ["x", "y"], "AB", transcripts, rows)
    assert context["kind"] == "judge_speech_conflict"
    rows[1]["parsed"].update(observable=False, winner="tie", gap="unobservable")
    with pytest.raises(ValueError, match="Unobservable"):
        protocol.judge_input(task(True), ["x", "y"], "AB", transcripts, rows)


def test_schema_adapts_handles_and_silent_audit():
    current = task()
    current["references"][0]["handle"] = "@portrait"
    assert protocol.schema("judge", current)["properties"]["per_reference"]["required"] == ["@portrait"]
    spec = protocol.schema("listener", current)
    assert spec["properties"]["per_reference"]["required"] == []
    value = response(spec)
    value["heard_content"].update(audio_state="no_audio_stream", transcript="", events=[])
    value["general_fidelity"]["audio_quality"].update(status="not_applicable", severity="none")
    protocol.validate_listener(value, current)
    value["heard_content"]["transcript"] = "invented words"
    with pytest.raises(ValueError):
        protocol.validate_listener(value, current)


def test_davidson_symmetric_ties_and_order_invariance():
    rows = []
    for a, b in [("a", "b"), ("a", "c"), ("b", "c")]:
        for winner in (a, b, "tie"):
            rows.append(
                {
                    "task_identity": digest([a, b, winner]),
                    "systems": [a, b],
                    "kind": "tie" if winner == "tie" else "decided",
                    "winner": winner,
                }
            )
    fit = ranking.fit(rows, ["a", "b", "c"], "conflict_as_tie", "davidson")
    assert np.allclose(list(fit["log_strength"].values()), 0, atol=1e-6)
    assert np.allclose(list(fit["expected_win_rate"].values()), 0.5, atol=1e-6)
    reordered = ranking.fit(list(reversed(rows)), ["c", "b", "a"], "conflict_as_tie", "davidson")
    assert fit["expected_win_rate"] == pytest.approx(reordered["expected_win_rate"])


def test_cache_keeps_one_response_and_never_records_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_URL", "https://example.invalid/v1/responses")
    monkeypatch.setenv("TEST_KEY", "test-secret-never-save")
    settings = {
        "api": "responses",
        "url_env": "TEST_URL",
        "api_key_env": "TEST_KEY",
        "model": "test-model",
        "reasoning_effort": "medium",
        "max_output_tokens": 8192,
        "max_attempts": 1,
    }
    client = Client(settings, tmp_path)
    sent = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                {"status": "completed", "output": [{"type": "message", "content": [{"text": '{"ok":true}'}]}]}
            ).encode()

    def send(request, timeout):
        sent.append(request.data)
        return Reply()

    monkeypatch.setattr(client.opener, "open", send)
    spec = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    for _ in range(2):
        result = client.generate("Judge this.", spec, [], lambda value: protocol.validate(value, spec))
        assert result["status"] == "ok"
    assert len(sent) == 1
    for path in tmp_path.rglob("*.json"):
        assert "test-secret-never-save" not in path.read_text()


def test_diagnostic_summary_requires_shared_support_and_sample_standardization(tmp_path):
    from types import SimpleNamespace

    from PIL import Image

    from orav_bench.diagnostics import summarize

    # Manifest loading only checks video path/type; no decoding is needed for aggregation.
    image = tmp_path / "ref.png"
    Image.new("RGB", (16, 16)).save(image)
    tasks = []
    for i in range(2):
        candidates = {}
        for system in ("a", "b"):
            path = tmp_path / f"{i}-{system}.mp4"
            path.write_bytes(f"{i}-{system}".encode())
            candidates[system] = str(path)
        tasks.append(
            {
                "id": str(i),
                "instruction": "Use @image1.",
                "references": [{"handle": "@image1", "role": "subject", "path": str(image)}],
                "candidates": candidates,
            }
        )
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, {"version": 1, "systems": ["a", "b"], "tasks": tasks})
    manifest = load_manifest(manifest_path)
    names = [
        "vbench_" + k
        for k in (
            "subject_consistency",
            "background_consistency",
            "motion_smoothness",
            "aesthetic_quality",
            "imaging_quality",
        )
    ]
    for i, current in enumerate(manifest["tasks"]):
        for j, system in enumerate(("a", "b")):
            row = {
                "family": "vbench",
                "configuration": "fixed",
                "task_identity": current["identity"],
                "system": system,
                "candidate_sha256": current["candidates"][system]["sha256"],
                "state": "ok",
                "scores": {name: float(i * 2 + j) for name in names},
            }
            write_json(tmp_path / "results" / "vbench" / f"{i}-{j}.json", row)
    output = tmp_path / "summary.json"
    summarize(SimpleNamespace(manifest=manifest_path, results=tmp_path / "results", output=output))
    summary = read_json(output)
    assert summary["metrics"]["comp_z"]["shared_tasks_n"] == 2
    assert summary["corpus_standardization"][names[0]]["std"] == pytest.approx(np.std([0, 1, 2, 3], ddof=1))


def test_non_json_response_is_terminal_on_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_URL", "https://example.invalid/v1/responses")
    client = Client(
        {
            "api": "responses",
            "url_env": "TEST_URL",
            "model": "test",
            "reasoning_effort": "medium",
            "max_output_tokens": 32,
        },
        tmp_path,
    )
    sent = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b"upstream returned an invalid envelope"

    def send(request, timeout):
        sent.append(request.data)
        return Reply()

    monkeypatch.setattr(client.opener, "open", send)
    for _ in range(2):
        assert client.generate("test", {"type": "object"}, [], lambda x: x)["status"] == "invalid"
    assert len(sent) == 1
