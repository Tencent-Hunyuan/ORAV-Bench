import copy
import json
from pathlib import Path

import pytest

from orav_bench import runner
from orav_bench.client import Client, endpoint_url, resolve_models
from orav_bench.io import read_json


def registry():
    return {
        "models": {
            "judge-model": {
                "label": "Judge display",
                "api": "responses",
                "model": "judge-deployment",
                "base_url_env": "TEST_JUDGE_BASE",
                "api_key_env": "TEST_JUDGE_KEY",
                "reasoning_effort": "medium",
                "max_output_tokens": 8192,
            },
            "listener-model": {
                "label": "Listener display",
                "api": "gemini",
                "model": "listener-deployment",
                "base_url_env": "TEST_LISTENER_BASE",
                "api_key_env": "TEST_LISTENER_KEY",
                "api_key_header": "x-goog-api-key",
                "temperature": 0.3,
                "reasoning_effort": "medium",
                "max_output_tokens": 32768,
            },
        },
        "judge": "judge-model",
        "listener": "listener-model",
    }


def test_independent_endpoints_authentication_and_model_selection(tmp_path, monkeypatch):
    config = registry()
    unchanged = copy.deepcopy(config)
    settings = resolve_models(config)
    monkeypatch.setenv("TEST_JUDGE_BASE", "https://judge.example/gateway/v1/")
    monkeypatch.setenv("TEST_LISTENER_BASE", "https://listener.example/v1beta")
    monkeypatch.setenv("TEST_JUDGE_KEY", "judge-secret")
    monkeypatch.setenv("TEST_LISTENER_KEY", "listener-secret")
    judge, listener = (Client(settings[role], tmp_path) for role in ("judge", "listener"))
    assert judge.url == "https://judge.example/gateway/v1/responses"
    assert listener.url == "https://listener.example/v1beta/models/listener-deployment:generateContent"
    assert judge.headers["Authorization"] == "Bearer judge-secret"
    assert listener.headers["x-goog-api-key"] == "listener-secret"
    assert "Authorization" not in listener.headers
    assert judge.body("test", {"type": "object"}, [])["model"] == "judge-deployment"
    assert "label" not in judge.body("test", {"type": "object"}, [])
    assert settings["judge"]["id"] == "judge-model"
    assert config == unchanged


def test_complete_endpoint_override_and_inline_configuration(tmp_path, monkeypatch):
    config = {
        role: dict(registry()["models"][model_id])
        for role, model_id in [("judge", "judge-model"), ("listener", "listener-model")]
    }
    for role in config:
        config[role].pop("base_url_env")
        config[role]["url_env"] = "TEST_" + role.upper() + "_URL"
    monkeypatch.setenv("TEST_JUDGE_URL", "https://judge.example/custom-route")
    monkeypatch.setenv("TEST_LISTENER_URL", "https://listener.example/custom-listen")
    assert resolve_models(config) == config
    assert Client(config["judge"], tmp_path).url == "https://judge.example/custom-route"
    assert Client(config["listener"], tmp_path).url == "https://listener.example/custom-listen"
    config["judge"]["base_url"] = "https://unused.example/v1"
    assert endpoint_url(config["judge"]) == "https://judge.example/custom-route"
    monkeypatch.delenv("TEST_JUDGE_URL")
    with pytest.raises(ValueError, match="TEST_JUDGE_URL"):
        endpoint_url(config["judge"])


def test_model_route_encoding_and_base_url_precedence(monkeypatch):
    config = {"api": "gemini", "model": "models/custom/model", "base_url": "https://host.example/v1beta/"}
    assert endpoint_url(config) == "https://host.example/v1beta/models/custom%2Fmodel:generateContent"
    config["base_url_env"] = "TEST_BASE"
    monkeypatch.setenv("TEST_BASE", "https://other.example/proxy/v1beta")
    assert endpoint_url(config).startswith("https://other.example/proxy/v1beta/models/")
    monkeypatch.delenv("TEST_BASE")
    with pytest.raises(ValueError, match="TEST_BASE"):
        endpoint_url(config)


@pytest.mark.parametrize(
    "base",
    [
        "http://remote.example/v1",
        "https://host.example/v1?key=secret",
        "https://user:secret@host.example/v1",
        "https://host.example/v1#fragment",
    ],
)
def test_invalid_base_urls_fail_without_exposing_their_values(base):
    with pytest.raises(ValueError) as error:
        endpoint_url({"api": "responses", "model": "test", "base_url": base})
    assert base not in str(error.value)


def test_label_changes_reuse_cache_but_endpoint_changes_do_not(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_JUDGE_BASE", "https://first.example/v1")
    settings = resolve_models(registry())["judge"]
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
        sent.append(request.full_url)
        return Reply()

    ids = []
    for label in ("First label", "Renamed label"):
        client = Client({**settings, "label": label}, tmp_path)
        monkeypatch.setattr(client.opener, "open", send)
        ids.append(client.generate("test", {"type": "object"}, [], lambda x: x)["request_identity"])
    assert len(sent) == 1 and ids[0] == ids[1]
    monkeypatch.setenv("TEST_JUDGE_BASE", "https://second.example/v1")
    client = Client(settings, tmp_path)
    monkeypatch.setattr(client.opener, "open", send)
    changed = client.generate("test", {"type": "object"}, [], lambda x: x)
    assert len(sent) == 2 and changed["request_identity"] != ids[0]


def test_label_and_unused_entries_preserve_run_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner.media, "prepare_task", lambda task, output: {"candidates": {}, "references": {}}
    )
    manifest = {"systems": ["a", "b"], "tasks": [{"identity": "task-id", "references": [], "candidates": {}}]}
    config = registry()
    runner.evaluate(manifest, config, tmp_path, prepare_only=True)
    original = read_json(tmp_path / "run.json")["signature"]
    config["models"]["judge-model"]["label"] = "New display label"
    config["models"]["unused"] = {"no_endpoint_or_credentials_needed": True}
    runner.evaluate(manifest, config, tmp_path, prepare_only=True)
    current = read_json(tmp_path / "run.json")
    assert current["signature"] == original
    assert current["settings"]["judge"]["label"] == "New display label"
    assert set(current["settings"]) == {"judge", "listener"}
    config["models"]["judge-model"]["model"] = "different-deployment"
    with pytest.raises(ValueError, match="different inputs/settings/source"):
        runner.evaluate(manifest, config, tmp_path, prepare_only=True)


def test_unknown_models_and_unsupported_listener_fail_early():
    config = registry()
    config["judge"] = "not-registered"
    with pytest.raises(ValueError, match="Unknown model ID"):
        resolve_models(config)
    config["judge"] = config["listener"] = "judge-model"
    with pytest.raises(ValueError, match="audio input support"):
        resolve_models(config)
    example = Path(__file__).parents[1] / "examples/evaluation.json"
    assert resolve_models(read_json(example))["judge"]["model"] == "gpt-6-astra"
