import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from vetscore.models import TextSegment, SourceExcerpt

templatesDir = Path(__file__).parent / "prompt_templates"
jinjaEnv = Environment(loader=FileSystemLoader(templatesDir))


decompositionPromptTemplate = jinjaEnv.get_template("decomposition_prompt.jinja")
verificationPromptTemplate = jinjaEnv.get_template("verification_prompt.jinja")
harmPotentialPromptTemplate = jinjaEnv.get_template("harm_potential_prompt.jinja")


def buildDecompositionPrompt(segment: TextSegment, contextSection: str) -> str:
    return decompositionPromptTemplate.render(
        context_section=contextSection.rstrip(),
        statement_text=segment.text,
    )


def verificationRenderContext(
    claims: list[str], sources: list[SourceExcerpt]
) -> dict:
    sourcesText = "\n\n---\n\n".join(
        f"[Source {i + 1}]\n{s.excerpt}"
        for i, s in enumerate(sources)
    )

    claimsText = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))

    return {
        "sources_text": sourcesText,
        "claims_text": claimsText,
        "claim_count": len(claims),
    }


def buildVerificationPrompt(
    claims: list[str], sources: list[SourceExcerpt]
) -> str:
    return verificationPromptTemplate.render(
        **verificationRenderContext(claims, sources),
        include_reasoning=True,
    )


def buildVerificationPromptNoReasoning(
    claims: list[str], sources: list[SourceExcerpt]
) -> str:
    """Same task as buildVerificationPrompt, verdict-only output."""
    return verificationPromptTemplate.render(
        **verificationRenderContext(claims, sources),
        include_reasoning=False,
    )


def harmPotentialRenderContext(
    claims: list[str], userQuestion: str | None
) -> dict:
    claimsText = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))

    questionSection = (
        f"**USER'S QUESTION:**\n{userQuestion}\n"
        if userQuestion
        else ""
    )

    return {
        "question_section": questionSection,
        "claims_text": claimsText,
        "claim_count": len(claims),
    }


def buildHarmPotentialPrompt(
    claims: list[str], userQuestion: str | None = None
) -> str:
    return harmPotentialPromptTemplate.render(
        **harmPotentialRenderContext(claims, userQuestion),
        include_reasoning=True,
    )


def buildHarmPotentialPromptNoReasoning(
    claims: list[str], userQuestion: str | None = None
) -> str:
    """Same rubric as buildHarmPotentialPrompt, score-only output."""
    return harmPotentialPromptTemplate.render(
        **harmPotentialRenderContext(claims, userQuestion),
        include_reasoning=False,
    )
