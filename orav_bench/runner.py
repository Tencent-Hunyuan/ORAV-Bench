"""Run the ORAV listener and judge, then preserve both ordered decisions."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from string import Template

from . import media, protocol
from .client import Client, cache_settings, resolve_models
from .io import digest, read_json, sha256_file, write_json


def _parallel(function, jobs, workers, stage):
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, *args): key for key, args in jobs.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as error:
                results[key] = {"status": "missing", "parsed": None, "error": str(error)[:500]}
            print(f"{stage}: {len(results)}/{len(jobs)}", flush=True)
    return results


def evaluate(manifest, settings, output, workers=8, prepare_only=False):
    settings = resolve_models(settings)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for task in manifest["tasks"]:
        protocol.route(task)
    code = {
        str(p.relative_to(Path(__file__).parent)): sha256_file(p)
        for p in Path(__file__).parent.rglob("*")
        if p.is_file() and p.suffix in (".py", ".txt", ".json")
    }
    signature = digest(
        {
            "tasks": [t["identity"] for t in manifest["tasks"]],
            "settings": {role: cache_settings(value) for role, value in settings.items()},
            "code": code,
        }
    )
    run = {
        "signature": signature,
        "systems": manifest["systems"],
        "tasks": manifest["tasks"],
        "settings": settings,
        "source_sha256": code,
    }
    run_path = output / "run.json"
    if run_path.exists() and read_json(run_path)["signature"] != signature:
        raise ValueError("Output belongs to different inputs/settings/source; choose a new output directory")
    if not prepare_only:
        # Resolve endpoints and credentials before expensive media preparation.
        judge = Client(settings["judge"], output)
        listener = (
            judge
            if cache_settings(settings["judge"]) == cache_settings(settings["listener"])
            else Client(settings["listener"], output)
        )
    write_json(run_path, run)
    tasks = {t["identity"]: t for t in manifest["tasks"]}
    prepared = _parallel(
        lambda t: media.prepare_task(t, output), {key: (t,) for key, t in tasks.items()}, workers, "prepare"
    )
    write_json(output / "prepared.json", prepared)
    if prepare_only:
        failures = sum("candidates" not in value for value in prepared.values())
        if failures:
            raise RuntimeError(
                f"Media preparation failed for {failures} task(s); see {output / 'prepared.json'}"
            )
        return prepared

    def listen(task, system):
        bundle = prepared[task["identity"]]
        if protocol.route(task) == "speech":
            audio = bundle["candidates"][system]["audio"]
            if audio is None:
                raise ValueError("No candidate audio for the speech listener")
            return listener.generate(
                protocol.PROMPTS["transcript"],
                protocol.schema("transcript"),
                [{**audio, "label": "AUDIO CANDIDATE"}],
                protocol.validate_transcript,
            )
        prompt = Template(protocol.PROMPTS["listener"]).substitute(task=protocol.task_text(task))
        return listener.generate(
            prompt,
            protocol.schema("listener", task),
            media.listener_attachments(task, bundle, system),
            lambda value: protocol.validate_listener(value, task),
        )

    audio_jobs = {
        (key, system): (task, system)
        for key, task in tasks.items()
        if "candidates" in prepared[key]
        for system in prepared[key]["candidates"]
    }
    heard = _parallel(listen, audio_jobs, workers, "listener")
    pairs = [
        (key, list(pair))
        for key, task in tasks.items()
        for pair in combinations(
            sorted(s for s, c in task["candidates"].items() if c["state"] == "delivered"), 2
        )
    ]

    def voice(task, systems, direction):
        bundle = prepared[task["identity"]]
        reference = next(r for r in task["references"] if r["semantic"] == "speech")
        records = [bundle["references"][reference["handle"]]["audio"]]
        records += [bundle["candidates"][system]["audio"] for system in systems]
        if any(record is None for record in records):
            raise ValueError("Missing voice-comparison audio")
        attachments = [
            {**record, "label": "AUDIO " + label + (" | @audio1" if label == "REFERENCE" else "")}
            for label, record in zip(("REFERENCE", "A", "B"), records)
        ]
        return {
            **listener.generate(
                protocol.PROMPTS["voice"], protocol.schema("voice"), attachments, protocol.validate_voice
            ),
            "systems": systems,
            "direction": direction,
        }

    voice_jobs = {
        (key, tuple(pair), direction): (
            tasks[key],
            pair if direction == "AB" else list(reversed(pair)),
            direction,
        )
        for key, pair in pairs
        if protocol.route(tasks[key]) == "speech" and "candidates" in prepared[key]
        for direction in ("AB", "BA")
    }
    voices = _parallel(voice, voice_jobs, workers, "voice")

    def compare(task, pair, direction):
        key = task["identity"]
        systems = pair if direction == "AB" else list(reversed(pair))
        row = {"systems": systems, "direction": direction}
        sources = [heard.get((key, system), {}) for system in systems]
        if any(source.get("status") != "ok" for source in sources):
            return {**row, "status": "missing", "parsed": None, "error": "Listener evidence unavailable"}
        voice_rows = None
        if protocol.route(task) == "speech":
            voice_rows = [voices.get((key, tuple(pair), dr), {}) for dr in ("AB", "BA")]
            if any(v.get("status") != "ok" for v in voice_rows):
                return {**row, "status": "missing", "parsed": None, "error": "Voice comparison unavailable"}
        try:
            prompt, schema, context = protocol.judge_input(
                task,
                systems,
                direction,
                {system: source["parsed"] for system, source in zip(systems, sources)},
                voice_rows,
            )
            result = judge.generate(
                prompt,
                schema,
                media.judge_attachments(task, prepared[key], systems),
                lambda value: protocol.validate_judge(value, task, context),
            )
        except (ValueError, KeyError) as error:
            result = {"status": "missing", "parsed": None, "error": str(error)}
        return {**row, **result}

    jobs = {
        (key, tuple(pair), direction): (tasks[key], pair, direction)
        for key, pair in pairs
        for direction in ("AB", "BA")
    }
    judged = _parallel(compare, jobs, workers, "judge")
    comparisons = []
    for key, pair in pairs:
        rows = [judged[(key, tuple(pair), direction)] for direction in ("AB", "BA")]
        comparisons.append(
            {
                "task_identity": key,
                "task_id": tasks[key]["key"],
                "systems": pair,
                "roles": tasks[key]["combination"],
                "directions": rows,
                "fused": protocol.fuse(rows, pair),
            }
        )
    result = {"signature": signature, "systems": manifest["systems"], "comparisons": comparisons}
    write_json(output / "comparisons.json", result)
    counts = dict(Counter(row["fused"]["status"] for row in comparisons))
    print(f"Comparison outcomes: {counts}; saved to {output / 'comparisons.json'}", flush=True)
    return result
