"""Configurable model APIs with immutable requests and resumable response caching."""

from __future__ import annotations

import base64
import fcntl
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlsplit

from .io import digest, read_json, sha256_file, write_json
from .protocol import compact


def resolve_models(config):
    """Resolve Judge and Listener settings from model IDs or inline configurations."""
    registry = config.get("models", {})
    if not isinstance(registry, dict):
        raise ValueError("models must map stable model IDs to configurations")
    selected = {}
    for role in ("judge", "listener"):
        value = config.get(role)
        if isinstance(value, str):
            if not value.strip() or value not in registry or not isinstance(registry[value], dict):
                raise ValueError(f"Unknown model ID for {role}: {value}")
            model_id = value
            value = {**registry[model_id], "id": model_id}
        elif isinstance(value, dict):
            value = dict(value)
        else:
            raise ValueError(f"{role} must name a model ID or contain inline model settings")
        if value.get("api") not in ("responses", "chat_completions", "gemini"):
            raise ValueError(f"{role}: api must be responses, chat_completions, or gemini")
        if role == "listener" and value["api"] != "gemini":
            raise ValueError("listener requires the gemini protocol with audio input support")
        if not isinstance(value.get("model"), str) or not value["model"].strip():
            raise ValueError(f"{role}: specify the actual API model name")
        if "label" in value and (not isinstance(value["label"], str) or not value["label"].strip()):
            raise ValueError(f"{role}: label must be a nonempty display name")
        selected[role] = value
    return selected


def cache_settings(settings):
    """Display names are metadata, not an input to the evaluator."""
    return {key: value for key, value in settings.items() if key != "label"}


def endpoint_url(settings):
    """Keep the configured API prefix; append only the protocol's resource path."""
    if settings.get("url_env"):
        # A full endpoint is an explicit override, including for custom gateways.
        url = os.environ.get(settings["url_env"], "").strip()
        if not url:
            raise ValueError(f"Set {settings['url_env']} to the complete API endpoint")
    else:
        if settings.get("base_url_env"):
            base = os.environ.get(settings["base_url_env"], "").strip()
            if not base:
                raise ValueError(f"Set {settings['base_url_env']} to the API base URL")
        else:
            base = settings.get("base_url", "").strip()
        if not base:
            raise ValueError("Configure base_url_env, base_url, or a complete endpoint through url_env")
        parsed = urlsplit(base)
        if parsed.query or parsed.fragment:
            raise ValueError(
                "A base URL must not contain a query or fragment; use url_env for a full endpoint"
            )
        if settings["api"] == "responses":
            suffix = "responses"
        elif settings["api"] == "chat_completions":
            suffix = "chat/completions"
        else:
            suffix = "models/" + quote(settings["model"].removeprefix("models/"), safe="") + ":generateContent"
        url = base.rstrip("/") + "/" + suffix
    parsed = urlsplit(url)
    if not parsed.hostname or not (
        parsed.scheme == "https"
        or (parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1"))
    ):
        raise ValueError("API endpoint must use HTTPS, or HTTP on loopback")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise ValueError(
            "Put credentials in API-key/header environment variables, not URL userinfo or fragments"
        )
    return url


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("API redirects are disabled to keep credentials at the configured endpoint")


class Client:
    def __init__(self, settings, output):
        if settings["api"] not in ("responses", "chat_completions", "gemini"):
            raise ValueError("API must be responses, chat_completions, or gemini")
        if settings["api"] == "chat_completions" and settings.get(
            "chat_token_limit_field", "max_completion_tokens"
        ) not in ("max_completion_tokens", "max_tokens"):
            raise ValueError("chat_token_limit_field must be max_completion_tokens or max_tokens")
        for key in ("requests_per_minute", "timeout_s", "max_attempts", "max_output_tokens"):
            if settings.get(key, 1) <= 0:
                raise ValueError(f"{key} must be positive")
        self.settings = settings
        self.output = Path(output)
        self._rate_lock = threading.Lock()
        self._next = 0.0
        self.url = endpoint_url(settings)
        self.headers = {"Content-Type": "application/json"}
        token = os.environ.get(settings.get("api_key_env", ""), "")
        if token:
            header = settings.get("api_key_header", "Authorization")
            self.headers[header] = ("Bearer " if header == "Authorization" else "") + token
        extra = os.environ.get(settings.get("headers_env", ""), "")
        if extra:
            self.headers.update(json.loads(extra))
        self._secrets = [str(v) for k, v in self.headers.items() if k != "Content-Type"]
        if token:
            self._secrets.append(token)
        self.opener = urllib.request.build_opener(NoRedirect())

    def _redact(self, value):
        if isinstance(value, str):
            for secret in sorted(self._secrets, key=len, reverse=True):
                if secret:
                    value = value.replace(secret, "[REDACTED]")
            return value.replace(self.url, "[API ENDPOINT]")
        if isinstance(value, dict):
            return {k: self._redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact(x) for x in value]
        return value

    def _throttle(self):
        rpm = self.settings.get("requests_per_minute", 25)
        with self._rate_lock:
            now = time.monotonic()
            delay = max(0, self._next - now)
            self._next = max(now, self._next) + 60 / rpm
        if delay:
            time.sleep(delay)

    def body(self, prompt, schema, attachments):
        settings = self.settings
        response_api = settings["api"] == "responses"
        chat_api = settings["api"] == "chat_completions"

        def text_part(text):
            if response_api:
                return {"type": "input_text", "text": text}
            if chat_api:
                return {"type": "text", "text": text}
            return {"text": text}

        parts = [text_part(prompt)]
        raw_size, images = 0, 0
        for item in attachments:
            label = f"[Attachment: {item['label']}]"
            parts.append(text_part(label))
            if "text" in item:
                parts.append(text_part(item["text"]))
                continue
            if sha256_file(item["path"]) != item["sha256"]:
                raise ValueError("Prepared media changed before request serialization")
            raw = Path(item["path"]).read_bytes()
            raw_size += len(raw)
            images += item["mime_type"].startswith("image/")
            encoded = base64.b64encode(raw).decode("ascii")
            if response_api or chat_api:
                if not item["mime_type"].startswith("image/"):
                    raise ValueError("The reasoning judge receives images and text only")
                url = f"data:{item['mime_type']};base64,{encoded}"
                if chat_api:
                    parts.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
                else:
                    parts.append({"type": "input_image", "image_url": url, "detail": "high"})
            else:
                parts.append({"inline_data": {"mime_type": item["mime_type"], "data": encoded}})
        if images > 50 or raw_size > 14_000_000:
            raise ValueError("Complete request exceeds the image/media budget; no evidence is dropped")
        if response_api:
            body = {
                "model": settings["model"],
                "input": [{"type": "message", "role": "user", "content": parts}],
                "max_output_tokens": settings["max_output_tokens"],
                "reasoning": {"effort": settings["reasoning_effort"]},
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "orav_result",
                        "schema": schema,
                        "strict": True,
                    }
                },
            }
        elif chat_api:
            body = {
                "model": settings["model"],
                "messages": [{"role": "user", "content": parts}],
                "n": 1,
                "stream": False,
                settings.get("chat_token_limit_field", "max_completion_tokens"): settings["max_output_tokens"],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "orav_result", "schema": schema, "strict": True},
                },
            }
            for key in ("reasoning_effort", "temperature"):
                if settings.get(key) is not None:
                    body[key] = settings[key]
        else:
            parts.append(
                {
                    "text": "Return exactly one JSON object conforming to this JSON Schema, with no prose outside it:\n"
                    + compact(schema)
                }
            )
            body = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": settings["temperature"],
                    "maxOutputTokens": settings["max_output_tokens"],
                    "responseMimeType": "application/json",
                    "thinkingConfig": {
                        "thinkingLevel": settings["reasoning_effort"],
                        "includeThoughts": False,
                    },
                },
            }
        if len(compact(body).encode()) > 20_000_000:
            raise ValueError("Serialized request exceeds 20 MB")
        return body

    def generate(self, prompt, schema, attachments, validator):
        public = {
            "settings": cache_settings(self.settings),
            "endpoint_digest": digest(self.url),
            "prompt": prompt,
            "schema": schema,
            "attachments": [{k: v for k, v in a.items() if k != "path"} for a in attachments],
        }
        identity = digest(public)
        home = self.output / "requests" / identity
        home.mkdir(parents=True, exist_ok=True)
        with (home / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result_path = home / "result.json"
            if result_path.exists():
                result = read_json(result_path)
                if result["request_identity"] != identity:
                    raise ValueError("Cache identity mismatch")
                if result["status"] == "ok":
                    validator(result["parsed"])
                return result
            body = self.body(prompt, schema, attachments)
            payload = compact(body).encode()
            write_json(home / "request.json", public)
            write_json(home / "payload.json", body)
            attempts = min(3, self.settings.get("max_attempts", 3))
            result = {"request_identity": identity, "status": "missing", "parsed": None}
            for attempt in range(attempts):
                receipt = home / f"attempt-{attempt}.json"
                if receipt.exists():
                    saved = read_json(receipt)
                    if saved.get("response") is not None:
                        envelope = saved["response"]
                        break
                    if not saved.get("retryable"):
                        return saved["result"]
                    continue
                self._throttle()
                request = urllib.request.Request(self.url, data=payload, headers=self.headers, method="POST")
                try:
                    with self.opener.open(request, timeout=self.settings.get("timeout_s", 600)) as response:
                        raw = response.read().decode("utf-8", errors="replace")
                    try:
                        envelope = json.loads(raw)
                    except ValueError:
                        result.update(status="invalid", error="API returned a non-JSON response")
                        write_json(
                            receipt, {"raw_response": self._redact(raw), "retryable": False, "result": result}
                        )
                        write_json(result_path, result)
                        return result
                    write_json(receipt, {"response": self._redact(envelope)})
                    break
                except (urllib.error.URLError, TimeoutError, OSError) as error:
                    status = getattr(error, "code", None)
                    retryable = status in (408, 429, 500, 502, 503, 504) or status is None
                    result["error"] = f"API request failed ({type(error).__name__}, HTTP {status})"
                    write_json(receipt, {"retryable": retryable, "result": result})
                    if not retryable or attempt + 1 == attempts:
                        write_json(result_path, result)
                        return result
                    time.sleep(min(2**attempt, 8))
            else:
                write_json(result_path, result)
                return result
            # Parse/semantic failures are terminal. No re-prompting or selection among valid answers.
            try:
                if self.settings["api"] == "responses":
                    text = "".join(
                        p.get("text", "")
                        for item in envelope.get("output", [])
                        if item.get("type") == "message"
                        for p in item.get("content", [])
                    )
                    if envelope.get("status") not in (None, "completed"):
                        raise ValueError("Incomplete API response")
                elif self.settings["api"] == "chat_completions":
                    choices = envelope.get("choices", [])
                    if not isinstance(choices, list) or len(choices) != 1:
                        raise ValueError("Expected exactly one chat completion")
                    choice = choices[0]
                    if choice.get("finish_reason") != "stop":
                        raise ValueError("Blocked or incomplete chat completion")
                    message = choice.get("message", {})
                    if message.get("refusal") or message.get("tool_calls") or message.get("function_call"):
                        raise ValueError("Chat completion returned a refusal or tool call")
                    text = message.get("content")
                    if not isinstance(text, str):
                        raise ValueError("Chat completion must contain a JSON text response")
                else:
                    candidates = envelope.get("candidates", [])
                    if not candidates or candidates[0].get("finishReason") not in (None, "STOP"):
                        raise ValueError("Blocked or incomplete listener response")
                    text = "".join(
                        p.get("text", "")
                        for p in candidates[0].get("content", {}).get("parts", [])
                        if not p.get("thought")
                    )
                text = text.strip()
                if text.startswith("```json") and text.endswith("```"):
                    text = text[7:-3].strip()
                parsed = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                validator(parsed)
                result.update(status="ok", parsed=parsed)
            except Exception as error:
                result.update(status="invalid", error=self._redact(str(error))[:1000])
            write_json(result_path, result)
            return result
