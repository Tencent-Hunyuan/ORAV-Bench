"""Evidence construction, response validation, and ordered-decision fusion."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from string import Template

from jsonschema import Draft202012Validator

from .io import read_json

HOME = Path(__file__).parent
SCHEMAS = read_json(HOME / "schemas.json")
PROMPTS = {p.stem: p.read_text() for p in (HOME / "prompts").glob("*.txt")}
REPLAY = re.compile(r"replay|reuse|reused|identical to the reference|same recording|played back", re.I)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def task_text(task):
    return json.dumps(
        {
            "generation_instruction": task["instruction_en"],
            "requested_duration_s": task.get("duration_s"),
            "reference_handles": [
                {k: r[k] for k in ("handle", "semantic", "modality")} for r in task["references"]
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def route(task):
    audio = [r for r in task["references"] if r["semantic"] in ("speech", "music")]
    speech = [r for r in audio if r["semantic"] == "speech"]
    if speech and (len(speech) != 1 or len(audio) != 1):
        raise ValueError("The speech protocol requires one speech reference and no other audio reference")
    return "speech" if speech else "standard"


def schema(kind, task=None):
    result = copy.deepcopy(SCHEMAS[kind])
    if "per_reference" in result["properties"]:
        refs = task["references"]
        if kind == "listener":
            refs = [r for r in refs if r["semantic"] in ("speech", "music")]
        field = result["properties"]["per_reference"]
        item = next(iter(field["properties"].values()))
        field["properties"] = {r["handle"]: copy.deepcopy(item) for r in refs}
        field["required"] = list(field["properties"])
    return result


def validate(value, specification):
    Draft202012Validator(specification).validate(value)


def _evidence_nonempty(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("evidence", "a_evidence", "b_evidence", "description") and not item.strip():
                raise ValueError("Empty evidence")
            _evidence_nonempty(item)
    elif isinstance(value, list):
        for item in value:
            _evidence_nonempty(item)


def validate_listener(value, task):
    validate(value, schema("listener", task))
    _evidence_nonempty(value)
    heard, quality = value["heard_content"], value["general_fidelity"]["audio_quality"]
    for event in heard["events"]:
        if event["end_s"] < event["start_s"]:
            raise ValueError("Reversed auditory event interval")
    absent = heard["audio_state"] in ("no_audio_stream", "decoded_silence")
    if absent and (heard["transcript"].strip() or heard["events"] or quality["status"] != "not_applicable"):
        raise ValueError("No-stream/silence report contains incompatible auditory claims")
    if quality["status"] != "assessed" and quality["severity"] != "none":
        raise ValueError("Unassessed quality cannot be a defect")
    for ref in value["per_reference"].values():
        if not ref["reference_observation"].strip() or not ref["candidate_observation"].strip():
            raise ValueError("Empty reference comparison")
        basis, status = ref["comparison_basis"], ref["status"]
        if (basis == "not_observable") != (status == "not_observable"):
            raise ValueError("Unobservable comparison cannot establish reference fidelity")
        if basis == "candidate_sound_absent" and (not absent or status == "supported"):
            raise ValueError("Sound absence cannot establish supported transfer")
        if basis == "direct_auditory_comparison" and heard["audio_state"] != "audible":
            raise ValueError("Direct comparison requires audible evidence")
    if len(compact(value).encode()) > 16000:
        raise ValueError("Auditory evidence exceeds the protocol limit")
    return value


def validate_voice(value):
    validate(value, schema("voice"))
    _evidence_nonempty(value)
    if not value["observable"]:
        if value["gap"] != "unobservable" or value["winner"] != "tie":
            raise ValueError("Unobservable voice evidence cannot yield a preference")
    elif value["gap"] == "unobservable" or any(
        c["similarity"] == "unobservable" for c in value["candidates"].values()
    ):
        raise ValueError("Observable comparison requires both voices")
    elif (value["gap"] == "none") != (value["winner"] == "tie"):
        raise ValueError("Voice gap and preference disagree")
    return value


def validate_transcript(value):
    validate(value, schema("transcript"))
    state, text = value["speech_state"], value["transcript"].strip()
    if not value["evidence"].strip():
        raise ValueError("Empty transcript evidence")
    if state in ("intelligible", "partial") and not text:
        raise ValueError("Discernible speech requires a transcript")
    if state in ("no_intelligible_speech", "not_observable") and text:
        raise ValueError("Unobserved speech cannot supply words")
    if state == "intelligible" and "[unclear]" in text:
        raise ValueError("Unclear words cannot be fully intelligible")
    if len(json.dumps(value, ensure_ascii=False).encode()) > 8000:
        raise ValueError("Transcript exceeds the protocol limit")
    return value


def public_audit(value, slot):
    def remap(item):
        if isinstance(item, str):
            return re.sub(r"(?<![A-Za-z0-9_])M0(?![A-Za-z0-9_])", slot, item)
        if isinstance(item, list):
            return [remap(x) for x in item]
        if isinstance(item, dict):
            return {k: remap(item[k]) for k in sorted(item)}
        return item

    return remap(value)


def judge_input(task, systems, direction, sound, voice_rows=None):
    facts = {slot: sound[system] for slot, system in zip("AB", systems)}
    if route(task) == "standard":
        facts = {slot: public_audit(value, slot) for slot, value in facts.items()}
        prompt = Template(PROMPTS["judge"]).substitute(task=task_text(task), sound=compact(facts))
        return prompt, schema("judge", task), {"kind": "judge"}
    if not voice_rows or {row["direction"] for row in voice_rows} != {"AB", "BA"}:
        raise ValueError("Two complete voice comparisons are required")
    mapped, replay = [], False
    for row in voice_rows:
        value = validate_voice(row["parsed"])
        if not value["observable"]:
            raise ValueError("Unobservable voice evidence is not an order conflict")
        mapped.append("tie" if value["winner"] == "tie" else row["systems"]["AB".index(value["winner"])])
        replay |= any(REPLAY.search(c["shared_and_different"]) for c in value["candidates"].values())
    same = next(row for row in voice_rows if row["direction"] == direction)
    if same["systems"] != systems or voice_rows[0]["systems"] != list(reversed(voice_rows[1]["systems"])):
        raise ValueError("Voice and visual candidate identities differ")
    kind = "judge_speech" if mapped[0] == mapped[1] else "judge_speech_conflict"
    prompt = Template(PROMPTS[kind]).substitute(
        task=task_text(task), content=compact(facts), voice=compact(same["parsed"])
    )
    if replay:
        prompt += "\n" + PROMPTS["replay_note"]
    return prompt, schema(kind, task), {"kind": kind, "voice": same["parsed"], "content": facts}


def validate_judge(value, task, context):
    validate(value, schema(context["kind"], task))
    _evidence_nonempty(value)
    for group in value["general_fidelity"].values():
        for item in group.values():
            if item["status"] != "assessed" and item["severity"] != "none":
                raise ValueError("Unassessed quality cannot be a defect")
    if context["kind"] == "judge":
        return value
    for slot, review in value["speech_content_review"].items():
        source = context["content"][slot]
        if review["heard_transcript"] != source["transcript"]:
            raise ValueError("Judge changed the listener transcript")
        quote, status = review["target_quote"], review["status"]
        if status == "not_applicable":
            if quote:
                raise ValueError("Inapplicable content has target words")
        elif not quote.strip() or quote not in task["instruction_en"]:
            raise ValueError("Target quote is absent from the generation instruction")
        if source["speech_state"] == "not_observable" and status not in ("not_observable", "not_applicable"):
            raise ValueError("Unknown content promoted to fact")
        if source["speech_state"] != "intelligible" and status == "supported":
            raise ValueError("Incomplete transcript cannot establish complete compliance")
    conflict = context["kind"] == "judge_speech_conflict"
    if value["sound_evidence_review"]["status"] != ("conflict" if conflict else "accepted"):
        raise ValueError("Judge did not retain the supplied auditory evidence state")
    for ref in task["references"]:
        if ref["semantic"] != "speech":
            continue
        item = value["per_reference"][ref["handle"]]
        expected = "tie" if conflict else context["voice"]["winner"]
        if item["winner"] != expected or item["evidence_sufficient"] == conflict:
            raise ValueError("Judge overwrote the supplied voice comparison")
    return value


def decision(value, systems):
    gates = [value["usability"][slot] for slot in "AB"]
    if all(g["evidence_sufficient"] for g in gates) and gates[0]["usable"] != gates[1]["usable"]:
        return {"status": "ok", "winner": systems[0 if gates[0]["usable"] else 1], "stage": "usability"}
    for field in ("overall_task", "quality_tiebreak"):
        item = value[field]
        if not item["evidence_sufficient"]:
            return {"status": "not_observable", "winner": None, "stage": field}
        if item["winner"] != "tie":
            return {"status": "ok", "winner": systems["AB".index(item["winner"])], "stage": field}
    return {"status": "ok", "winner": "tie", "stage": "task_and_quality_tie"}


def fuse(rows, systems):
    if (
        len(rows) != 2
        or {r["direction"] for r in rows} != {"AB", "BA"}
        or any(r["status"] != "ok" for r in rows)
    ):
        return {"status": "missing", "winner": None}
    for row in rows:
        expected = systems if row["direction"] == "AB" else list(reversed(systems))
        if row["systems"] != expected:
            raise ValueError("Candidate slot mapping changed")
    choices = [decision(row["parsed"], row["systems"]) for row in rows]
    if any(c["status"] != "ok" for c in choices):
        return {"status": "not_observable", "winner": None, "directions": choices}
    if choices[0]["winner"] != choices[1]["winner"]:
        return {"status": "order_conflict", "winner": None, "directions": choices}
    return {"status": "ok", "winner": choices[0]["winner"], "directions": choices}
