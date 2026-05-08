"""OpenAI-compatible LLM client with an offline-safe wrapper.

The rest of the system never imports the `openai` package directly; it
goes through `LLMClient`. If `OPENAI_API_KEY` is unset, or the openai
package is not installed, or any call raises, `available()` returns
False and the methods return empty strings. The agent loop interprets
that as "skip the LLM step."

Configuration via env vars:
- OPENAI_API_KEY     — required for any call
- OPENAI_BASE_URL    — optional, for OpenAI-compatible endpoints
- OPENAI_MODEL       — defaults to gpt-4o-mini
- OPENAI_TIMEOUT_S   — defaults to 30
- OPENAI_MAX_TOKENS  — defaults to 1024
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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

    def propose_fix(self, system_prompt: str, source: str, test_output: str) -> str:
        """Ask the model for a corrected solution.py. Empty string on failure."""
        if not self.available():
            return ""
        sys_msg = system_prompt or (
            "You are a precise Python bug-fixer. The user shows you a single-function "
            "module that has one small bug, plus the test output. Reply with ONLY the "
            "full corrected source of solution.py — no fences, no commentary."
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
            text = (resp.choices[0].message.content or "").strip()
            return _strip_fence(text)
        except Exception:
            return ""

    def summarize_bug_families(self, samples: List[Tuple[str, str]]) -> str:
        """Given (task_id, buggy_source) samples, return a short hint string.

        Returns "" on failure or when unavailable.
        """
        if not self.available() or not samples:
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
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""
