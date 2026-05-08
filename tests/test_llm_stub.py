"""Stub-injected LLMClient tests.

The offline tests already cover the no-key path. This file exercises the
*available* path without a real network call by injecting a fake openai
client onto an `LLMClient` instance, and verifies:

- `_strip_fence` strips bare and language-tagged code fences;
- a successful proposal is inserted at the head of the agent's variant
  queue (so it is tried first, with `fixing_primitive == "llm_proposal"`);
- a recognized error class raised inside the SDK is recorded on
  `errors_by_type` / `last_error_type`, and the agent surfaces the
  class name on `SolveResult.note`.
"""
from pathlib import Path

import pytest

from stem_agent.agent import solve_task
from stem_agent.blueprint import PRIMITIVE_NAMES, Blueprint
from stem_agent.llm import LLMClient, LLMConfig, _strip_fence


# ---------- _strip_fence -------------------------------------------------


def test_strip_fence_no_fence_passes_through():
    src = "def f():\n    return 1\n"
    assert _strip_fence(src) == src.strip()


def test_strip_fence_plain_fence():
    text = "```\ndef f():\n    return 1\n```\n"
    assert _strip_fence(text) == "def f():\n    return 1"


def test_strip_fence_language_tag_fence():
    text = "```python\ndef f():\n    return 1\n```\n"
    assert _strip_fence(text) == "def f():\n    return 1"


def test_strip_fence_with_surrounding_whitespace():
    text = "  \n```python\ncode\n```\n  "
    assert _strip_fence(text) == "code"


# ---------- fake openai client ------------------------------------------


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, content=None, raises=None) -> None:
        self._content = content
        self._raises = raises
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return _FakeCompletion(self._content)


class _FakeChat:
    def __init__(self, content=None, raises=None) -> None:
        self.completions = _FakeChatCompletions(content=content, raises=raises)


class _FakeOpenAI:
    def __init__(self, content=None, raises=None) -> None:
        self.chat = _FakeChat(content=content, raises=raises)


def _make_stub_client(*, content=None, raises=None) -> LLMClient:
    """Return an `LLMClient` with a fake openai client wired into it.

    Sidesteps the real `_init_client` (which needs the openai package
    and a key) so the test runs offline. `available()` returns True.
    """
    c = LLMClient(config=LLMConfig(api_key="dummy", model="x"))
    c._client = _FakeOpenAI(content=content, raises=raises)
    c._init_error = None
    return c


# ---------- propose_fix --------------------------------------------------


def test_stub_propose_fix_returns_stripped_text():
    c = _make_stub_client(content="```python\ndef g():\n    return 2\n```")
    out = c.propose_fix("", "def g(): return 1", "")
    assert out == "def g():\n    return 2"
    assert c.calls == 1
    assert c.errors_by_type == {}
    assert c.last_error_type() is None


def test_stub_propose_fix_records_recognized_error():
    c = _make_stub_client(raises=OSError("simulated"))
    out = c.propose_fix("", "def g(): return 1", "")
    assert out == ""
    assert c.calls == 1
    assert c.errors_by_type == {"OSError": 1}
    assert c.last_error_type() == "OSError"


def test_summary_string_reflects_call_history():
    c = _make_stub_client(content="def f(): return 0\n")
    c.propose_fix("", "x", "")
    assert "1 calls" in c.summary()
    assert "0 errors" in c.summary()

    c2 = _make_stub_client(raises=OSError("net"))
    c2.propose_fix("", "x", "")
    s = c2.summary()
    assert "OSError=1" in s
    assert "1 errors" in s


# ---------- agent integration -------------------------------------------


def _make_task(parent: Path, sol: str, test_src: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "solution.py").write_text(sol, encoding="utf-8")
    (parent / "test_solution.py").write_text(test_src, encoding="utf-8")
    return parent


def test_stub_proposal_inserted_at_index_0(tmp_path):
    """The LLM's proposal is tried before any deterministic variant.

    The buggy source has no deterministic fix in the primitive bank
    (no constant +/- 1 reaches 42), so the only way the task can be
    solved at iteration 1 is if the proposal lives at index 0 of the
    variant queue.
    """
    td = _make_task(
        tmp_path / "task",
        "def f():\n    return 0\n",
        "from solution import f\n\n\ndef test_a():\n    assert f() == 42\n",
    )
    fixed = "def f():\n    return 42\n"
    c = _make_stub_client(content=fixed)
    bp = Blueprint(
        name="llm-bp",
        primitive_priority=list(PRIMITIVE_NAMES),
        primitive_budget=8,
        early_stop_no_progress=8,
        use_llm_proposal=True,
    )
    res = solve_task(td, bp, llm=c, task_id="task")
    assert res.solved is True
    assert res.fixing_primitive == "llm_proposal"
    assert res.iterations == 1


def test_stub_error_propagates_to_solve_result_note(tmp_path):
    """When the LLM raises a recognized error, the class lands on the note.

    The deterministic primitives still solve the task (here, the
    `flip_compare` primitive), so `solved` is True; the LLM failure is
    additive metadata rather than a fatal error.
    """
    td = _make_task(
        tmp_path / "task",
        "def f(x):\n    return x < 1\n",
        "from solution import f\n\n\ndef test_a():\n    assert f(2) is True\n",
    )
    c = _make_stub_client(raises=OSError("network"))
    bp = Blueprint(
        name="llm-bp",
        primitive_priority=list(PRIMITIVE_NAMES),
        primitive_budget=8,
        early_stop_no_progress=8,
        use_llm_proposal=True,
    )
    res = solve_task(td, bp, llm=c, task_id="task")
    assert res.solved is True
    assert "llm: OSError" in res.note
    assert c.last_error_type() == "OSError"
