"""OpenAI-compatible LLM client with an offline-safe wrapper.

The rest of the system never imports the `openai` package directly; it
goes through `LLMClient`. If `OPENAI_API_KEY` is unset, or the openai
package is not installed, `available()` returns False and the methods
return empty strings. The agent loop interprets that as "skip the LLM
step."

When the client is available but a call fails, the failure is attributed
to a recognized error class (auth, rate limit, network, etc.) rather
than swallowed: the per-instance `calls` and `errors_by_type` counters
record what happened, `last_error_type()` exposes the most recent
failure class, and the agent surfaces it on `SolveResult.note` so a
silent regression cannot masquerade as the deterministic-path numbers.

Configuration via env vars:
- OPENAI_API_KEY     required for any call
- OPENAI_BASE_URL    optional, for OpenAI-compatible endpoints
- OPENAI_MODEL       defaults to gpt-4o-mini
- OPENAI_TIMEOUT_S   defaults to 30
- OPENAI_MAX_TOKENS  defaults to 1024
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _resolve_call_error_classes() -> Tuple[type, ...]:
    """Return the tuple of exception classes that count as recoverable LLM-call errors.

    We catch the openai SDK's own base error class (covers auth, rate
    limit, timeout, connection, server-side, bad-request, etc.), the
    httpx transport errors that the SDK can re-raise during streaming
    or eager requests, and OSError as a last-mile network/socket
    fallback. Anything outside this set is allowed to propagate so a
    real bug in the agent doesn't get silently misattributed.
    """
    classes: List[type] = []
    try:
        import openai  # type: ignore

        classes.append(openai.OpenAIError)
    except (ImportError, AttributeError):
        pass
    try:
        import httpx  # type: ignore

        classes.append(httpx.HTTPError)
    except (ImportError, AttributeError):
        pass
    classes.append(OSError)
    return tuple(classes)


_CALL_ERROR_CLASSES: Tuple[type, ...] = _resolve_call_error_classes()


@dataclass
class LLMConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4o-mini"
    timeout_s: float = 30.0
    max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY") or None,
            base_url=os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or None,
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            timeout_s=float(os.environ.get("OPENAI_TIMEOUT_S", "30")),
            max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "1024")),
        )


def _strip_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


class LLMClient:
    """Best-effort wrapper around an OpenAI-compatible chat-completions API."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._client = None
        self._init_error: Optional[str] = None
        self.calls: int = 0
        self.errors_by_type: Dict[str, int] = {}
        self._last_error_type: Optional[str] = None
        self._init_client()

    def _init_client(self) -> None:
        if not self.config.api_key:
            self._init_error = "no api key"
            return
        try:
            from openai import OpenAI
        except ImportError:
            self._init_error = "openai package not installed"
            return
        try:
            kwargs = {"api_key": self.config.api_key}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**kwargs)
        except Exception as e:
            self._init_error = f"init: {type(e).__name__}: {e}"

    def available(self) -> bool:
        return self._client is not None

    def status(self) -> str:
        return "available" if self.available() else (self._init_error or "unavailable")

    def last_error_type(self) -> Optional[str]:
        """Class name of the most recent call's error, or None on success."""
        return self._last_error_type

    def summary(self) -> str:
        """One-line human-readable summary suitable for CLI output."""
        if not self.available() and self.calls == 0:
            return f"llm: {self.status()}"
        if self.calls == 0:
            return "llm: 0 calls"
        n_errors = sum(self.errors_by_type.values())
        n_ok = self.calls - n_errors
        if not self.errors_by_type:
            return f"llm: {self.calls} calls, {n_ok} ok, 0 errors"
        breakdown = ", ".join(
            f"{cls}={n}" for cls, n in sorted(self.errors_by_type.items())
        )
        return (
            f"llm: {self.calls} calls, {n_ok} ok, {n_errors} errors ({breakdown})"
        )

    def _record_success(self) -> None:
        self.calls += 1
        self._last_error_type = None

    def _record_error(self, exc: BaseException) -> None:
        self.calls += 1
        name = type(exc).__name__
        self.errors_by_type[name] = self.errors_by_type.get(name, 0) + 1
        self._last_error_type = name

    def propose_fix(self, system_prompt: str, source: str, test_output: str) -> str:
        """Ask the model for a corrected solution.py. Empty string on failure.

        Failures are recorded on `errors_by_type` and exposed via
        `last_error_type()`; the agent surfaces the class name onto
        `SolveResult.note`.
        """
        if not self.available():
            self._last_error_type = None
            return ""
        sys_msg = system_prompt or (
            "You are a precise Python bug-fixer. The user shows you a single-function "
            "module that has one small bug, plus the test output. Reply with ONLY the "
            "full corrected source of solution.py: no fences, no commentary."
        )
        user = (
            "----- solution.py (buggy) -----\n"
            f"{source}\n"
            "----- test output (truncated) -----\n"
            f"{test_output[-2000:]}\n"
            "----- end -----\n"
            "Reply with ONLY the corrected solution.py."
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout_s,
            )
        except _CALL_ERROR_CLASSES as e:
            self._record_error(e)
            return ""
        self._record_success()
        text = (resp.choices[0].message.content or "").strip()
        return _strip_fence(text)

    def summarize_bug_families(self, samples: List[Tuple[str, str]]) -> str:
        """Given (task_id, buggy_source) samples, return a short hint string.

        Returns "" on failure or when unavailable.
        """
        if not self.available() or not samples:
            self._last_error_type = None
            return ""
        joined = "\n\n".join(f"# {tid}\n{src.strip()}" for tid, src in samples[:8])
        try:
            resp = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You concisely summarize Python bug families.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Below are buggy Python single-function modules. Reply with EXACTLY "
                            "3 short bullet lines summarizing recurring bug types in this batch. "
                            "No preamble. No closing.\n\n" + joined
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=256,
                timeout=self.config.timeout_s,
            )
        except _CALL_ERROR_CLASSES as e:
            self._record_error(e)
            return ""
        self._record_success()
        return (resp.choices[0].message.content or "").strip()
