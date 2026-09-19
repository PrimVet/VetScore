import logging
from collections.abc import Callable
from dataclasses import dataclass

from vetscore import config
from vetscore.llm_backends import call_ollama_json, call_openrouter_json, infer_backend
from vetscore.models import (
    INVALID_ITEM,
    NO_SOURCES,
    NOT_RETURNED,
    SourceExcerpt,
    TextSegment,
    TokenUsage,
    injected_reason,
    openrouter_response_format,
    parse_results_payload,
)
from vetscore.prompts import buildVerificationPrompt, buildVerificationPromptNoReasoning

logger = logging.getLogger(__name__)


@dataclass
class VerifiedClaim:
    claim: str
    is_supported: bool | None
    reasoning: str
    supported_fraction: float | None = None
    verdicts: list[bool] | None = None
    reasonings: list[str] | None = None


def combine_verdict_samples(
    claims: list[str],
    samples: list[list[VerifiedClaim]],
    with_aggregates: bool = True,
) -> list[VerifiedClaim]:
    results: list[VerifiedClaim] = []
    for i, claim in enumerate(claims):
        voting = [
            sample[i]
            for sample in samples
            if i < len(sample) and sample[i].is_supported is not None
        ]
        if not voting:
            results.append(
                VerifiedClaim(claim=claim, is_supported=None, reasoning=NOT_RETURNED)
            )
            continue
        verdicts = [v.is_supported for v in voting]
        reasonings = [v.reasoning for v in voting]
        fraction = sum(1 for v in verdicts if v) / len(verdicts)
        results.append(
            VerifiedClaim(
                claim=claim,
                is_supported=fraction > 0.5,
                reasoning=next((r for r in reasonings if r), ""),
                supported_fraction=fraction if with_aggregates else None,
                verdicts=verdicts if with_aggregates else None,
                reasonings=reasonings if with_aggregates else None,
            )
        )
    return results


class Verifier:
    """Verifies atomic claims against source excerpts."""

    def __init__(
        self,
        client,
        model: str,
        include_reasoning: bool = True,
        temperature: float = 1.0,
        max_retries: int = 3,
        reasoning_level: str = "none",
        schema_root: str = "array",
        provider: dict | None = None,
        num_samples: int = 1,
    ):
        self.client = client
        self.backend = infer_backend(client)
        self.model = model
        self.include_reasoning = include_reasoning
        self.temperature = temperature
        self.max_retries = max_retries
        self.reasoning_level = reasoning_level
        self.schema_root = schema_root
        self.provider = provider
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        self.num_samples = num_samples

    def __call__(
        self,
        segments: list[TextSegment],
        segment_claims: list[list[str]],
        on_segment: Callable[[int, list], None] | None = None,
    ) -> list[list[VerifiedClaim]]:
        results: list[list[VerifiedClaim]] = []
        for i, (seg, claims) in enumerate(zip(segments, segment_claims)):
            verified, _usage = self.verify_segment(claims, seg.sources, i)
            results.append(verified)

            supported = sum(1 for v in verified if v.is_supported)
            logger.info(
                "Verified segment %d/%d: %d/%d supported",
                i + 1,
                len(segments),
                supported,
                len(verified),
            )
            if on_segment is not None:
                on_segment(len(results), results)

        return results

    def verify_segment(
        self, claims: list[str], sources: list[SourceExcerpt], segment_index: int
    ) -> tuple[list[VerifiedClaim], TokenUsage]:
        if not claims:
            return [], TokenUsage()

        if not sources:
            return [
                VerifiedClaim(claim=c, is_supported=False, reasoning=NO_SOURCES)
                for c in claims
            ], TokenUsage()

        prompt = (
            buildVerificationPrompt(claims, sources)
            if self.include_reasoning
            else buildVerificationPromptNoReasoning(claims, sources)
        )

        total_usage = TokenUsage()
        samples: list[list[VerifiedClaim]] = []
        for i in range(self.num_samples):
            label = f" for segment {segment_index}"
            if self.num_samples > 1:
                label += f" (sample {i + 1}/{self.num_samples})"
            verified, usage = self._draw_verdict_sample(prompt, label, segment_index)
            total_usage = total_usage + usage
            if verified is not None:
                samples.append(verified)

        if not samples:
            reason = injected_reason(
                f"Failed after {self.max_retries} attempts"
                if self.num_samples == 1
                else f"All {self.num_samples} samples failed"
            )
            return [
                VerifiedClaim(claim=c, is_supported=None, reasoning=reason)
                for c in claims
            ], total_usage
        return (
            combine_verdict_samples(claims, samples, self.num_samples > 1),
            total_usage,
        )

    def _draw_verdict_sample(
        self, prompt: str, context_label: str, segment_index: int
    ) -> tuple[list[VerifiedClaim] | None, TokenUsage]:
        if self.backend in ("openrouter", "ollama"):
            item_properties: dict[str, dict] = {"claim": {"type": "string"}}
            if self.include_reasoning:
                item_properties["reasoning"] = {"type": "string"}
            item_properties["isSupported"] = {"type": "boolean"}
            response_format = openrouter_response_format(
                "verification_results", item_properties, self.schema_root
            )
            if self.backend == "openrouter":
                extra_body: dict = {"reasoning": config.OPENROUTER_REASONING[self.reasoning_level]}
                if self.provider:
                    extra_body["provider"] = self.provider

                parsed, usage = call_openrouter_json(
                    self.client,
                    model=self.model,
                    prompt=prompt,
                    response_format=response_format,
                    extra_body=extra_body,
                    temperature=self.temperature,
                    max_retries=self.max_retries,
                    max_output_tokens=config.MAX_OUTPUT_TOKENS,
                    unwrap=parse_results_payload,
                    context_label=context_label,
                )
            else:
                parsed, usage = call_ollama_json(
                    self.client,
                    model=self.model,
                    prompt=prompt,
                    response_format=response_format["json_schema"]["schema"],
                    think=config.OLLAMA_THINK[self.reasoning_level],
                    temperature=self.temperature,
                    max_retries=self.max_retries,
                    max_output_tokens=config.MAX_OUTPUT_TOKENS,
                    unwrap=parse_results_payload,
                    context_label=context_label,
                )
            if parsed is None:
                return None, usage

            results_out = []
            for item in parsed:
                if not isinstance(item, dict):
                    logger.warning(
                        "Non-dict item in results for segment %d: %r", segment_index, item
                    )
                    results_out.append(
                        VerifiedClaim(claim="", is_supported=None, reasoning=INVALID_ITEM)
                    )
                    continue
                results_out.append(VerifiedClaim(
                    claim=item.get("claim", ""),
                    is_supported=bool(item.get("isSupported")) if item.get("isSupported") is not None else None,
                    reasoning=item.get("reasoning", ""),
                ))
            return results_out, usage

        else:
            raise ValueError(f"unknown backend {self.backend!r}")
