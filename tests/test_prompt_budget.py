"""Generic context selection budgets."""
from app.services.prompts import build_rag_prompt


def test_invalid_and_duplicate_passages_do_not_waste_budget():
    prompt = build_rag_prompt("q", [
        {"text": None}, {"text": 42}, {"text": "a" * 100},
        {"text": " aaaaa "}, {"text": "aaaaa"}, {"text": "bbbbb"},
    ], max_context_chars=10)
    assert "[1] aaaaa" in prompt and "[2] bbbbb" in prompt
    assert "a" * 100 not in prompt
