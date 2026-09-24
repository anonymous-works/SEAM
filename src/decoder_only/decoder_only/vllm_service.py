from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


def normalize_base_url(value: str) -> str:

    raw = str(value).strip().rstrip("/")
    if not raw:
        raise ValueError("vLLM base URL cannot be empty")
    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"vLLM base URL must be an HTTP(S) URL, got {value!r}")
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


class VLLMClient:
    def __init__(self, base_url: str, *, api_key: str = "EMPTY", timeout: float = 900.0) -> None:
        self.base_url = normalize_base_url(base_url)
        self.api_key = str(api_key)
        self.timeout = float(timeout)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key and self.api_key.upper() != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(_url(self.base_url, path), data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {exc.code} at {path}: {detail[:1000]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach vLLM at {self.base_url}: {exc.reason}") from exc
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"vLLM returned invalid JSON from {path}: {raw[:500]!r}") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError(f"vLLM returned a non-object response from {path}")
        return decoded

    def models(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/models")
        models = response.get("data")
        if not isinstance(models, list) or not models:
            raise RuntimeError(f"vLLM /models returned no served model: {response!r}")
        valid_models = [item for item in models if isinstance(item, dict)]
        if not valid_models:
            raise RuntimeError(f"vLLM /models returned malformed model entries: {response!r}")
        return valid_models

    def model_name(self, requested: str | None = None) -> str:
        if requested and str(requested).strip():
            return str(requested).strip()
        models = self.models()
        name = models[0].get("id")
        if not str(name or "").strip():
            raise RuntimeError(f"vLLM served model has no id: {models[0]!r}")
        return str(name)

    def wait_until_ready(
        self,
        *,
        timeout_seconds: float,
        poll_seconds: float = 2.0,
        process: subprocess.Popen[str] | None = None,
    ) -> str:

        deadline = time.monotonic() + float(timeout_seconds)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"vLLM server exited with status {process.returncode} before becoming ready")
            try:
                return self.model_name()
            except (RuntimeError, URLError, OSError) as exc:
                last_error = exc
                time.sleep(min(float(poll_seconds), max(0.05, deadline - time.monotonic())))
        raise TimeoutError(f"Timed out after {timeout_seconds:.1f}s waiting for vLLM at {self.base_url}: {last_error}")

    def complete(
        self,
        prompt_ids: list[list[int]],
        generation: dict[str, Any],
        *,
        model: str,
    ) -> list[str]:

        if not prompt_ids:
            return []
        if any(not prompt for prompt in prompt_ids):
            raise ValueError("vLLM prompts must be non-empty token-id lists")
        do_sample = bool(generation.get("do_sample", False))
        temperature = float(generation.get("temperature", 0.0))
        if do_sample and temperature <= 0:
            raise ValueError("Sampling requires generation.temperature > 0")
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt_ids[0] if len(prompt_ids) == 1 else prompt_ids,
            "max_tokens": int(generation["max_new_tokens"]),
            "min_tokens": int(generation.get("min_new_tokens", 0)),
            "temperature": temperature if do_sample else 0.0,
            "top_p": float(generation.get("top_p", 1.0)),
            "n": 1,
            "stream": False,
        }

        top_k = int(generation.get("top_k", 0))
        if top_k > 0:
            payload["top_k"] = top_k
        repetition_penalty = float(generation.get("repetition_penalty", 1.0))
        if repetition_penalty != 1.0:
            payload["repetition_penalty"] = repetition_penalty

        response = self._request("POST", "/completions", payload)
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != len(prompt_ids):
            raise RuntimeError(
                f"Expected {len(prompt_ids)} vLLM completion choices, got {len(choices) if isinstance(choices, list) else choices!r}"
            )
        ordered: list[str | None] = [None] * len(prompt_ids)
        for fallback_index, choice in enumerate(choices):
            if not isinstance(choice, dict):
                raise RuntimeError(f"Malformed vLLM completion choice: {choice!r}")
            index = int(choice.get("index", fallback_index))
            if index < 0 or index >= len(ordered) or ordered[index] is not None:
                raise RuntimeError(f"Malformed vLLM choice index {index}: {choices!r}")
            ordered[index] = str(choice.get("text", "")).strip()
        if any(value is None for value in ordered):
            raise RuntimeError(f"vLLM did not return every completion: {choices!r}")
        return [str(value) for value in ordered]


def _local_host(host: str) -> bool:
    return host in {"localhost", "127.0.0.1", "::1"}


class VLLMServer:
    def __init__(self, process: subprocess.Popen[str], log_handle: Any, log_path: Path, client: VLLMClient) -> None:
        self.process = process
        self.log_handle = log_handle
        self.log_path = log_path
        self.client = client

    @classmethod
    def start(
        cls,
        checkpoint: str | Path,
        *,
        base_url: str,
        max_model_len: int,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        startup_timeout: float = 900.0,
        log_path: str | Path | None = None,
    ) -> "VLLMServer":
        client = VLLMClient(
            base_url,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            timeout=max(30.0, float(startup_timeout)),
        )
        parts = urlsplit(client.base_url)
        host = parts.hostname or "127.0.0.1"
        if not _local_host(host):
            raise ValueError(f"--start-vllm-service only accepts a local URL, got host {host!r}")
        port = parts.port or 8000
        command = [
            os.environ.get("VLLM_BIN", "vllm"),
            "serve",
            str(Path(checkpoint).expanduser().resolve()),
            "--host",
            host,
            "--port",
            str(port),
            "--dtype",
            str(dtype),
            "--max-model-len",
            str(int(max_model_len)),
            "--generation-config",
            "vllm",
        ]
        if trust_remote_code:
            command.append("--trust-remote-code")
        extra = os.environ.get("VLLM_SERVER_EXTRA_ARGS", "").strip()
        if extra:
            command.extend(shlex.split(extra))
        resolved_log = Path(log_path).expanduser().resolve() if log_path else Path("vllm_server.log").resolve()
        resolved_log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = resolved_log.open("a", encoding="utf-8")
        log_handle.write(f"\n[start] {' '.join(shlex.quote(item) for item in command)}\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        server = cls(process, log_handle, resolved_log, client)
        try:
            model = client.wait_until_ready(timeout_seconds=startup_timeout, process=process)
        except Exception:
            server.stop()
            raise RuntimeError(f"vLLM failed to start; inspect {resolved_log}") from None
        print(f"[vllm] service ready: url={client.base_url} model={model} log={resolved_log}", flush=True)
        return server

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log_handle.close()

    def __enter__(self) -> "VLLMServer":
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


def iter_batches(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), int(batch_size)):
        yield values[start : start + int(batch_size)]
