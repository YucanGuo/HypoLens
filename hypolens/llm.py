"""
LLMClient
=========
Unified chat client supporting two backends:

  1. vllm (local)  — OpenAI-compatible server, no API key required.
     Auto-detects the running model via GET /v1/models.

  2. Remote API    — Any OpenAI-compatible provider (DeepSeek, OpenAI, etc.)
     Requires an API key.  Pass api_key or set LLM_API_KEY env var.

Backend selection
-----------------
If `api_key` is provided (or LLM_API_KEY is set), the remote-API mode is used.
Otherwise, the client connects to the local vllm server.

Environment variables
---------------------
LLM_BASE_URL    : server base URL
                  · vllm default  : http://localhost:8000/v1
                  · DeepSeek      : https://api.deepseek.com/v1
                  · OpenAI        : https://api.openai.com/v1
LLM_API_KEY     : API key (leave unset for local vllm)
LLM_MODEL       : model name (auto-detected for vllm; required for remote APIs
                  unless passed explicitly, e.g. "deepseek-chat")
LLM_TIMEOUT     : request timeout in seconds  (default: 120)
LLM_MAX_TOKENS  : default max_tokens          (default: 16384)
LLM_TEMPERATURE : default sampling temperature (default: 0.0)

Quick examples
--------------
# Local vllm (auto-detect model)
client = LLMClient()

# DeepSeek API
client = LLMClient(
    api_key  = "sk-...",
    base_url = "https://api.deepseek.com/v1",
    model    = "deepseek-chat",
)

# Via environment variables
export LLM_API_KEY=sk-...
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_MODEL=deepseek-chat
client = LLMClient()
"""

from __future__ import annotations

import os
from typing import Any

import httpx

# ── defaults ───────────────────────────────────────────────────────────────────

_BASE_URL    = os.getenv("LLM_BASE_URL",    "http://localhost:8000/v1")
_API_KEY     = os.getenv("LLM_API_KEY",     "")          # empty → local vllm
_TIMEOUT     = float(os.getenv("LLM_TIMEOUT",     "120"))
_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS",    "16384"))
_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))


# ── message helpers ────────────────────────────────────────────────────────────

def system_msg(content: str) -> dict[str, str]:
    return {"role": "system", "content": content}

def user_msg(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}

def assistant_msg(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content}


# ── client ─────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Synchronous HTTP client for any OpenAI-compatible chat completions endpoint.

    Parameters
    ----------
    model : str | None
        Model name.
        · vllm    : auto-detected from /v1/models if None.
        · Remote  : must be set (e.g. "deepseek-chat", "gpt-4o").
          Falls back to LLM_MODEL env var.
    api_key : str | None
        API key for remote providers.  If None, falls back to LLM_API_KEY env
        var.  Empty string / not set → local vllm (no Authorization header).
    base_url : str
        Base URL of the API server (no trailing slash).
    timeout : float
        HTTP request timeout (seconds).
    max_tokens : int
        Default token budget for each generation call.
    temperature : float
        Default sampling temperature.
    """

    def __init__(
        self,
        model:       str | None = None,
        api_key:     str | None = None,
        base_url:    str        = _BASE_URL,
        timeout:     float      = _TIMEOUT,
        max_tokens:  int        = _MAX_TOKENS,
        temperature: float      = _TEMPERATURE,
    ) -> None:
        self.base_url    = base_url.rstrip("/")
        self.timeout     = timeout
        self.max_tokens  = max_tokens
        self.temperature = temperature

        # Resolve API key (param > env > empty)
        resolved_key = api_key if api_key is not None else _API_KEY

        # Build headers
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if resolved_key:
            headers["Authorization"] = f"Bearer {resolved_key}"
        self._is_remote = bool(resolved_key)

        self._http = httpx.Client(timeout=self.timeout, headers=headers)

        # Resolve model name
        if model:
            self.model = model
        else:
            env_model = os.getenv("LLM_MODEL", "")
            if env_model:
                self.model = env_model
            elif not self._is_remote:
                # Local vllm: auto-detect
                self.model = self._detect_model()
            else:
                raise ValueError(
                    "model must be specified for remote API usage "
                    "(or set the LLM_MODEL environment variable)."
                )

    # ── public API ─────────────────────────────────────────────────────────────

    def chat(
        self,
        messages:    list[dict[str, str]],
        max_tokens:  int   | None = None,
        temperature: float | None = None,
        stop:        list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Send a list of chat messages and return the assistant reply as a string.

        Parameters
        ----------
        messages : list[dict]
            List of {"role": ..., "content": ...} dicts.
        max_tokens : int | None
            Override the instance default.
        temperature : float | None
            Override the instance default.
        stop : list[str] | None
            Optional stop sequences.

        Returns
        -------
        str
            The assistant's reply content.

        Raises
        ------
        httpx.HTTPStatusError
            If the server returns a non-2xx response.
        """
        payload: dict[str, Any] = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  max_tokens  if max_tokens  is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)

        resp = self._http.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def complete(
        self,
        prompt:      str,
        max_tokens:  int   | None = None,
        temperature: float | None = None,
        stop:        list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Convenience wrapper: wrap *prompt* in a single user message."""
        return self.chat(
            [user_msg(prompt)],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs,
        )

    @property
    def backend(self) -> str:
        """Human-readable backend description."""
        return f"remote({self.base_url})" if self._is_remote else f"vllm({self.base_url})"

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"LLMClient(model={self.model!r}, backend={self.backend!r}, "
            f"max_tokens={self.max_tokens})"
        )

    # ── internal ───────────────────────────────────────────────────────────────

    def _detect_model(self) -> str:
        """
        Query /v1/models and return the id of the first available model.
        Used only for local vllm (no API key).
        """
        try:
            resp = self._http.get(f"{self.base_url}/models")
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if not models:
                raise RuntimeError("No models registered in the vllm server.")
            return models[0]["id"]
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot reach vllm server at {self.base_url}. "
                f"Start it with: vllm serve <model> --port 8000"
            ) from exc
