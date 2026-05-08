"""LLM client must be safe with no API key, no network access."""

from stem_agent.llm import LLMClient, LLMConfig


def test_llm_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    c = LLMClient()
    assert c.available() is False
    assert c.propose_fix("", "x = 1", "") == ""
    assert c.summarize_bug_families([("a", "x = 1")]) == ""


def test_llm_status_message_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = LLMClient()
    msg = c.status()
    assert "no api key" in msg or "openai" in msg


def test_llm_explicit_empty_config():
    c = LLMClient(config=LLMConfig(api_key=None, model="x"))
    assert c.available() is False
    assert c.propose_fix("", "", "") == ""
