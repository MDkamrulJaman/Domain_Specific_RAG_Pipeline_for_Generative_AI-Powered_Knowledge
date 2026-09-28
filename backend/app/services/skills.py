"""Immutable NVIDIA task policies; no network calls or shared request state."""
from dataclasses import dataclass
from typing import Literal

SkillName = Literal["general", "summarize", "explain", "requirements"]


@dataclass(frozen=True)
class Skill:
    label: str
    instruction: str
    max_tokens: int | None = None
    max_context_chars: int | None = None
    retrieval_top_k: int = 5


# Data-driven Strategy policies. Frozen values are safe to reuse across requests;
# sharing this small registry is not a full GoF Flyweight implementation.
SKILLS: dict[SkillName, Skill] = {
    "general": Skill("General answer", "", max_context_chars=16000),
    "summarize": Skill(
        "Summarize retrieved passages",
        "Summarize the retrieved passages relevant to the question in at most five concise bullets. "
        "Preserve qualifications and limitations. Explicitly state that this summary covers only "
        "retrieved passages, not the entire document. Cite passage labels such as [1].",
        512, 12000, retrieval_top_k=8,
    ),
    "explain": Skill(
        "Explain a technical concept",
        "Explain the requested concept in plain language, followed by the key technical details. "
        "Include one example only if supported by the passages. Cite passage labels such as [1].",
        768, 16000, retrieval_top_k=4,
    ),
    "requirements": Skill(
        "Find requirements",
        "Extract only requirements relevant to the question. Preserve their exact wording, "
        "identifiers, conditions, and mandatory versus optional language. Cite each requirement "
        "using its passage label, such as [1]. Do not invent requirements or source locations. "
        "If none are present, say so. Do not claim the list covers the entire document.",
        1024, 20000, retrieval_top_k=8,
    ),
}


def get_skill(name: SkillName) -> Skill:
    try:
        return SKILLS[name]
    except KeyError:
        raise ValueError(f"Unknown NVIDIA skill: {name}") from None
