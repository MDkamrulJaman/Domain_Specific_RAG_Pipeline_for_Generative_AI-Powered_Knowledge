"""Inactive reference only. Not imported or used by the application."""
from dataclasses import dataclass
from typing import Literal, Mapping
from types import MappingProxyType

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
SKILLS: Mapping[SkillName, Skill] = MappingProxyType({
    "general": Skill("General answer", "", max_context_chars=16000),
    "summarize": Skill(
        "Summarize retrieved passages",
        "Summarize relevant evidence in up to five bullets with citations. "
        "Cover retrieved passages, not the entire document.",
        512, 12000, retrieval_top_k=8,
    ),
    "explain": Skill(
        "Explain a technical concept",
        "Explain clearly using the evidence and cite sources. Include an example only if supported.",
        768, 16000, retrieval_top_k=4,
    ),
    "requirements": Skill(
        "Find requirements",
        "List relevant requirements with citations. Preserve exact wording, identifiers and conditions. "
        "If absent, say so; do not claim complete document coverage.",
        1024, 20000, retrieval_top_k=8,
    ),
})


def get_skill(name: SkillName) -> Skill:
    try:
        return SKILLS[name]
    except KeyError:
        raise ValueError(f"Unknown NVIDIA skill: {name}") from None
