import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from vetscore import config
from vetscore.llm_backends import call_ollama_json, call_openrouter_json, infer_backend
from vetscore.models import (
    DEFAULT_ON_ERROR,
    NOT_RETURNED,
    TokenUsage,
    openrouter_response_format,
    parse_results_payload,
)
from vetscore.prompts import (
    buildHarmPotentialPrompt,
    buildHarmPotentialPromptNoReasoning,
)

logger = logging.getLogger(__name__)


@dataclass
class HarmPotentialResult:
    score: int | None
    reasoning: str
    score_mean: float | None = None
    scores: list[int] | None = None
    reasonings: list[str] | None = None


def combine_score_samples(
    claims: list[str],
    samples: list[dict[str, HarmPotentialResult]],
    with_aggregates: bool = True,
) -> list[HarmPotentialResult]:
    results: list[HarmPotentialResult] = []
    for claim in claims:
        scored = [
            m[claim] for m in samples if claim in m and m[claim].score is not None
        ]
        if not scored:
            results.append(
                HarmPotentialResult(score=None, reasoning=NOT_RETURNED)
            )
            continue
        scores = [r.score for r in scored]
        mean = sum(scores) / len(scores)
        reasonings = [r.reasoning for r in scored]
        results.append(
            HarmPotentialResult(
                score=max(1, min(5, math.floor(mean + 0.5))),
                reasoning=next((r for r in reasonings if r), ""),
                score_mean=mean if with_aggregates else None,
                scores=scores if with_aggregates else None,
                reasonings=reasonings if with_aggregates else None,
            )
        )
    return results


def default_results(claims: list[str]) -> list[HarmPotentialResult]:
    return [
        HarmPotentialResult(
            score=None,
            reasoning=DEFAULT_ON_ERROR,
        )
        for _ in claims
    ]


class HarmPotentialScorer:
    """Scores atomic claims for harm potential (1-5)."""

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
        segment_claims: list[list[str]],
        user_question: str | Sequence[str | None] | None = None,
        on_segment: Callable[[int, list], None] | None = None,
    ) -> list[list[HarmPotentialResult]]:
        if user_question is None or isinstance(user_question, str):
            questions: list[str | None] = [user_question] * len(segment_claims)
        else:
            questions = list(user_question)
            if len(questions) != len(segment_claims):
                raise ValueError(
                    f"user_question has {len(questions)} entries but there are "
                    f"{len(segment_claims)} segments to score."
                )

        results: list[list[HarmPotentialResult]] = []
        for i, claims in enumerate(segment_claims):
            if not claims:
                results.append([])
                if on_segment is not None:
                    on_segment(len(results), results)
                continue
            res, _usage = self.score_segment(claims, i, questions[i])
            results.append(res)

            logger.info(
                "Scored harm potential for segment %d/%d (%d claims)",
                i + 1,
                len(segment_claims),
                len(claims),
            )
            if on_segment is not None:
                on_segment(len(results), results)

        return results

    def score_segment(
        self, claims: list[str], segment_index: int, user_question: str | None = None
    ) -> tuple[list[HarmPotentialResult], TokenUsage]:
        if not claims:
            return [], TokenUsage()

        prompt = (
            buildHarmPotentialPrompt(claims, user_question)
            if self.include_reasoning
            else buildHarmPotentialPromptNoReasoning(claims, user_question)
        )

        total_usage = TokenUsage()
        samples = []
        for i in range(self.num_samples):
            label = f" for segment {segment_index}"
            if self.num_samples > 1:
                label += f" (sample {i + 1}/{self.num_samples})"
            result_map, usage = self._draw_score_sample(prompt, label)
            total_usage = total_usage + usage
            if result_map is not None:
                samples.append(result_map)

        if not samples:
            return default_results(claims), total_usage
        return (
            combine_score_samples(claims, samples, self.num_samples > 1),
            total_usage,
        )

    def _draw_score_sample(
        self, prompt: str, context_label: str
    ) -> tuple[dict[str, HarmPotentialResult] | None, TokenUsage]:
        if self.backend in ("openrouter", "ollama"):
            item_properties: dict[str, dict] = {"claim": {"type": "string"}}
            if self.include_reasoning:
                item_properties["reasoning"] = {"type": "string"}
            item_properties["score"] = {"type": "integer"}
            response_format = openrouter_response_format(
                "scoring_results", item_properties, self.schema_root
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

            result_map: dict[str, HarmPotentialResult] = {}
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                claim_text = item.get("claim", "")
                if not claim_text:
                    continue
                raw_score = item.get("score")
                clamped = max(1, min(5, int(raw_score))) if raw_score is not None else None
                result_map[claim_text] = HarmPotentialResult(
                    score=clamped,
                    reasoning=item.get("reasoning", ""),
                )

            return result_map, usage

        else:
            raise ValueError(f"unknown backend {self.backend!r}")
