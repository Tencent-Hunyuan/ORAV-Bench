import base64
import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from orav_bench import protocol
from orav_bench.client import Client, endpoint_url, resolve_models
from orav_bench.io import read_json, sha256_file

SPEC = {
    "type": "object",
    "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie"]}},
    "required": ["winner"],
    "additionalProperties": False,
}
VALID_CHOICE = {
    "index": 0,
    "finish_reason": "stop",
    "message": {"role": "assistant", "content": '{"winner":"A"}'},
}


def settings(**changes):
    return {
        "api": "chat_completions",
        "model": "vision-judge",
        "base_url": "https://judge.example/gateway/v1/",
        "max_output_tokens": 8192,
        **changes,
    }


def test_chat_preserves_schema_and_ordered_multimodal_evidence(tmp_path):
    panel = tmp_path / "panel.jpg"
    Image.new("RGB", (8, 8), "green").save(panel)
    attachments = [
        {"label": "Listener evidence", "text": "Candidate A says hello."},
        {"label": "A_P000", "path": str(panel), "sha256": sha256_file(panel), "mime_type": "image/jpeg"},
        {"label": "Timing", "text": "Frame timestamps span the full clip."},
    ]
    task = {"references": [{"handle": "@image1", "semantic": "subject", "modality": "image"}]}
    schema = protocol.schema("judge", task)
    chat = Client(settings(), tmp_path).body("Complete prompt.", schema, attachments)
    responses = Client(settings(api="responses", reasoning_effort="medium"), tmp_path).body(
        "Complete prompt.", schema, attachments
    )
    parts = chat["messages"][0]["content"]
    original = responses["input"][0]["content"]
    assert [p.get("text") for p in parts] == [p.get("text") for p in original]
    assert parts[4]["image_url"] == {"url": original[4]["image_url"], "detail": "high"}
    assert base64.b64decode(parts[4]["image_url"]["url"].split(",", 1)[1]) == panel.read_bytes()
    assert chat["response_format"]["json_schema"]["schema"] == responses["text"]["format"]["schema"]
    assert chat["response_format"]["json_schema"]["strict"] is True
    assert chat["n"] == 1 and chat["stream"] is False
    with pytest.raises(ValueError, match="images and text only"):
        Client(settings(), tmp_path).body("test", schema, [{**attachments[1], "mime_type": "audio/wav"}])


def test_chat_configuration_and_explicit_parameter_compatibility(tmp_path, monkeypatch):
    config = read_json(Path(__file__).parents[1] / "examples/evaluation.json")
    config["judge"] = "chat-judge"
    selected = resolve_models(config)
    monkeypatch.setenv("ORAV_CHAT_BASE_URL", "https://judge.example/proxy/v1/")
    client = Client(selected["judge"], tmp_path)
    assert client.url == "https://judge.example/proxy/v1/chat/completions"
    body = client.body("test", SPEC, [])
    assert body["max_completion_tokens"] == 8192
    assert not {"reasoning_effort", "temperature", "max_tokens"} & body.keys()
    custom = Client(
        settings(chat_token_limit_field="max_tokens", reasoning_effort="low", temperature=0), tmp_path
    ).body("test", SPEC, [])
    assert custom["max_tokens"] == 8192 and "max_completion_tokens" not in custom
    assert custom["reasoning_effort"] == "low" and custom["temperature"] == 0
    with pytest.raises(ValueError, match="chat_token_limit_field"):
        Client(settings(chat_token_limit_field="maxOutputTokens"), tmp_path)
    monkeypatch.setenv("CHAT_FULL_URL", "https://judge.example/custom-chat")
    assert endpoint_url(settings(url_env="CHAT_FULL_URL")) == "https://judge.example/custom-chat"
    config["listener"] = "chat-judge"
    with pytest.raises(ValueError, match="audio input support"):
        resolve_models(config)


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("valid", "ok"),
        ("length", "invalid"),
        ("content_filter", "invalid"),
        ("no_finish_reason", "invalid"),
        ("refusal", "invalid"),
        ("tool_calls", "invalid"),
        ("multiple_choices", "invalid"),
        ("reasoning_only", "invalid"),
        ("invalid_schema", "invalid"),
    ],
)
def test_chat_responses_validate_and_remain_terminal_on_resume(tmp_path, monkeypatch, variant, expected):
    choice = copy.deepcopy(VALID_CHOICE)
    if variant in ("length", "content_filter"):
        choice["finish_reason"] = variant
    elif variant == "no_finish_reason":
        choice.pop("finish_reason")
    elif variant == "refusal":
        choice["message"]["refusal"] = "Cannot evaluate."
    elif variant == "tool_calls":
        choice["message"]["tool_calls"] = [{"type": "function"}]
    elif variant == "invalid_schema":
        choice["message"]["content"] = '{"winner":"unknown"}'
    # A separate reasoning field is never used as the final answer.
    choice["message"]["reasoning_content"] = '{"winner":"B"}'
    if variant == "reasoning_only":
        choice["message"]["content"] = None
    envelope = {"choices": [choice, choice] if variant == "multiple_choices" else [choice]}
    calls = []
    monkeypatch.setenv("CHAT_TEST_KEY", "chat-secret-never-save")
    client = Client(settings(api_key_env="CHAT_TEST_KEY"), tmp_path)

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(envelope).encode()

    def send(request, timeout):
        assert request.headers["Authorization"] == "Bearer chat-secret-never-save"
        calls.append(request.data)
        return Reply()

    monkeypatch.setattr(client.opener, "open", send)
    for _ in range(2):
        result = client.generate("test", SPEC, [], lambda value: protocol.validate(value, SPEC))
        assert result["status"] == expected
        if expected == "ok":
            assert result["parsed"] == {"winner": "A"}
    assert len(calls) == 1
    for path in tmp_path.rglob("*.json"):
        assert "chat-secret-never-save" not in path.read_text()
