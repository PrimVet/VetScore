import logging
from collections.abc import Callable

from vetscore import config
from vetscore.claim_decomposer import ClaimDecomposer
from vetscore.harm_potential import HarmPotentialScorer
from vetscore.models import (
    AtomicClaimOutput,
    SegmentOutput,
    SegmentVerificationStats,
    TextSegment,
    VerificationInput,
    VerificationOutput,
    VerificationSummary,
)
from vetscore.scoring import WeightedScoring
from vetscore.text_segmentation import TextSegmentation
from vetscore.verifier import Verifier

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        text_segmentation: TextSegmentation | None = None,
        decomposer: ClaimDecomposer | None = None,
        harm_scorer: HarmPotentialScorer | None = None,
        verifier: Verifier | None = None,
        scorer: WeightedScoring | None = None,
    ):
        self.text_segmentation = text_segmentation
        self.decomposer = decomposer
        self.harm_scorer = harm_scorer
        self.verifier = verifier
        self.scorer = scorer or WeightedScoring()

    def __call__(
        self,
        inp: VerificationInput,
        sample_segments: int | None = None,
        fixed_segment_indices: list[int] | None = None,
        sample_seed: int | None = None,
        precomputed_segments: list[TextSegment] | None = None,
        precomputed_sampled_indices: list[int] | None = None,
        precomputed_claims: list[list[str]] | None = None,
        plain_text: str | None = None,
        independent_segments: bool = False,
        user_questions: list[str | None] | None = None,
        checkpoint_every: int | None = None,
        on_checkpoint: Callable[[VerificationOutput], None] | None = None,
    ) -> VerificationOutput:
        parsed_plain_text = None
        if precomputed_segments is not None:
            logger.info("Step 1: Using %d precomputed segments", len(precomputed_segments))
            segments = precomputed_segments
        elif self.text_segmentation is not None:
            logger.info("Step 1: Segmenting synthesis content...")
            segments, parsed_plain_text = self.text_segmentation.segment_with_plain_text(inp)
            logger.info("Segmented into %d segments", len(segments))
        else:
            raise ValueError(
                "Pipeline has no text_segmentation configured and no "
                "precomputed_segments were provided."
            )

        if plain_text is None:
            if parsed_plain_text is not None:
                plain_text = parsed_plain_text
            elif self.text_segmentation is not None:
                plain_text = self.text_segmentation.get_plain_text(inp.synthesis_content)

        def has_real_citations(seg: TextSegment) -> bool:
            return bool(seg.sources)

        cited_indices = [i for i, seg in enumerate(segments) if has_real_citations(seg)]
        cited_segments = [segments[i] for i in cited_indices]
        logger.info(
            "Segments with citations: %d/%d", len(cited_segments), len(segments)
        )

        # Optional: use precomputed sampled indices, fixed indices, or random sample
        if precomputed_sampled_indices is not None:
            cited_indices = [i for i in precomputed_sampled_indices if i in cited_indices]
            cited_segments = [segments[i] for i in cited_indices]
            logger.info(
                "Using %d precomputed sampled indices", len(cited_segments)
            )
        elif fixed_segment_indices is not None:
            cited_indices = [i for i in fixed_segment_indices if i in cited_indices]
            cited_segments = [segments[i] for i in cited_indices]
            logger.info(
                "Using %d fixed segment indices", len(cited_segments)
            )
        elif sample_segments is not None and sample_segments < len(cited_indices):
            import random
            rng = random.Random(sample_seed)
            sampled = sorted(rng.sample(range(len(cited_indices)), sample_segments))
            cited_indices = [cited_indices[j] for j in sampled]
            cited_segments = [segments[i] for i in cited_indices]
            logger.info(
                "Sampled %d/%d cited segments", len(cited_segments), len(segments)
            )

        # Step 2: decompose cited segments into atomic claims (or use precomputed claims)
        if precomputed_claims is not None:
            logger.info("Step 2: Using precomputed claims")
            cited_decomposition = [
                (precomputed_claims[idx], None) for idx in cited_indices
            ]
        elif self.decomposer is not None:
            logger.info("Step 2: Decomposing segments into atomic claims...")
            cited_decomposition = []
            for j, idx in enumerate(cited_indices):
                seg = segments[idx]
                # Build context from plain_text up to segment's start_offset
                if independent_segments:
                    ctx_text = ""  # no preceding text: the records are unrelated
                elif plain_text is not None and seg.start_offset is not None:
                    ctx_text = plain_text[:seg.start_offset]
                else:
                    ctx_text = None  # fall back to previous_segments in decomposer
                previous = segments[:idx] if ctx_text is None else []
                claims, ctx = self.decomposer.decompose_segment(
                    seg, previous, idx, context_text=ctx_text,
                )
                cited_decomposition.append((claims, ctx))
                total = sum(len(r[0]) for r in cited_decomposition)
                logger.info(
                    "Decomposed segment %d/%d (idx=%d) → %d claims (total: %d)",
                    j + 1, len(cited_indices), idx, len(claims), total,
                )
        else:
            raise ValueError(
                "Pipeline has no decomposer configured and no precomputed_claims "
                "were provided."
            )

        cited_claims = [r[0] for r in cited_decomposition]

        n_cited = len(cited_indices)
        can_score = self.verifier is not None

        def _build_output_from_cited(
            cited_harm, cited_verif, done_cited: int | None = None
        ) -> VerificationOutput:
            harm = list(cited_harm) + [[]] * (n_cited - len(cited_harm))
            verif = list(cited_verif) + [[]] * (n_cited - len(cited_verif))

            position = {idx: j for j, idx in enumerate(cited_indices)}
            decomposition_results = []
            harm_potential_results = []
            verification_results = []
            for i in range(len(segments)):
                j = position.get(i)
                if j is None:
                    decomposition_results.append(([], None))
                    harm_potential_results.append([])
                    verification_results.append([])
                else:
                    decomposition_results.append(cited_decomposition[j])
                    harm_potential_results.append(harm[j])
                    verification_results.append(verif[j])

            end = (
                len(segments)
                if done_cited is None
                else cited_indices[done_cited - 1] + 1
            )
            return self._build_output(
                segments[:end],
                decomposition_results[:end],
                harm_potential_results[:end],
                verification_results[:end],
                can_score=can_score,
            )

        def _checkpoint_kwargs(slot) -> dict:
            if not checkpoint_every or on_checkpoint is None:
                return {}

            def _checkpoint_if_due(done: int, results: list) -> None:
                # The last segment needs no checkpoint — the full output is
                # written immediately afterwards.
                if done % checkpoint_every or done >= n_cited:
                    return
                logger.info("Checkpoint at segment %d/%d", done, n_cited)
                on_checkpoint(_build_output_from_cited(*slot(results), done))

            return {"on_segment": _checkpoint_if_due}

        # Step 3: score harm potential
        if self.harm_scorer is not None:
            logger.info("Step 3: Scoring harm potential...")
            if user_questions is not None:
                questions: str | list[str | None] | None = [
                    user_questions[idx] for idx in cited_indices
                ]
            else:
                questions = inp.user_question

            cited_harm_potential = self.harm_scorer(
                cited_claims,
                questions,
                **({} if can_score else _checkpoint_kwargs(lambda r: (r, []))),
            )
        else:
            logger.warning(
                "Step 3: No harm_scorer configured — every claim will use the "
                "default harm potential (%d); the weighted score will not "
                "reflect real harm-potential input.",
                config.DEFAULT_HARM_SCORE,
            )
            cited_harm_potential = [[None] * len(c) for c in cited_claims]

        # Step 4: verify claims against excerpts
        if self.verifier is not None:
            logger.info("Step 4: Verifying claims against source excerpts...")
            cited_verification = self.verifier(
                cited_segments,
                cited_claims,
                **_checkpoint_kwargs(lambda r: (cited_harm_potential, r)),
            )
        else:
            logger.warning(
                "Step 4: No verifier configured — claims cannot be verified, "
                "so no verification score can be computed."
            )
            cited_verification = [[None] * len(c) for c in cited_claims]

        # Step 5: build output
        logger.info("Step 5: Building output...")
        return _build_output_from_cited(cited_harm_potential, cited_verification)

    def _build_output(
        self,
        segments: list[TextSegment],
        decomposition_results: list[tuple[list[str], str | None]],
        harm_potential_results,
        verification_results,
        can_score: bool,
    ) -> VerificationOutput:
        segment_outputs: list[SegmentOutput] = []
        all_claims_for_summary: list[dict] = []

        for i, seg in enumerate(segments):
            claims, decomp_ctx = decomposition_results[i]
            harm_potentials = harm_potential_results[i]
            verifications = verification_results[i]

            # Build atomic claim outputs
            atomic_claims: list[AtomicClaimOutput] = []
            claims_for_scoring: list[dict] = []

            for j, claim_text in enumerate(claims):
                # Get harm potential (default DEFAULT_HARM_SCORE if missing)
                hp = harm_potentials[j] if j < len(harm_potentials) else None
                hp_score = hp.score if hp else config.DEFAULT_HARM_SCORE
                hp_reasoning = hp.reasoning if hp else None
                hp_samples = getattr(hp, "scores", None)
                hp_mean = getattr(hp, "score_mean", None)

                # Get verification (unknown if no verifier was configured,
                # otherwise default unsupported if missing)
                if can_score:
                    ver = verifications[j] if j < len(verifications) else None
                    is_supported = ver.is_supported if ver else False
                    reasoning = ver.reasoning if ver else "No verification result"
                    supported_fraction = ver.supported_fraction if ver else None
                    verdicts = ver.verdicts if ver else None
                    ver_reasonings = ver.reasonings if ver else None
                else:
                    is_supported = None
                    reasoning = "No verifier configured"
                    supported_fraction = None
                    verdicts = None
                    ver_reasonings = None

                if hp_samples:
                    hp_reported = hp_mean if hp_mean is not None else hp_score
                    hp_reasoning = None
                else:
                    hp_reported = hp_score
                supported_reported = (
                    supported_fraction
                    if verdicts and supported_fraction is not None
                    else is_supported
                )

                atomic_claims.append(
                    AtomicClaimOutput(
                        text=claim_text,
                        harm_potential=hp_reported,
                        harm_potential_samples=hp_samples,
                        harm_potential_reasoning=hp_reasoning,
                        harm_potential_reasoning_samples=getattr(
                            hp, "reasonings", None
                        ),
                        is_supported=supported_reported,
                        is_supported_samples=verdicts,
                        is_supported_reasoning=reasoning,
                        is_supported_reasoning_samples=ver_reasonings,
                    )
                )

                claim_entry = {
                    "is_supported": is_supported,
                    "harm_potential": hp_reported,
                }
                claims_for_scoring.append(claim_entry)
                all_claims_for_summary.append(claim_entry)

            # Compute segment-level scores
            total = len(claims)
            supported = sum(1 for c in claims_for_scoring if c["is_supported"])
            if can_score:
                raw_score = supported / total if total > 0 else 0.0
                weighted = self.scorer(claims_for_scoring)
            else:
                raw_score = None
                weighted = None

            # Build excerpt list for output
            excerpts = [s.model_dump() for s in seg.sources]

            segment_outputs.append(
                SegmentOutput(
                    segment_index=i,
                    text=seg.text,
                    decomposition_context=decomp_ctx,
                    atomic_claims=atomic_claims,
                    excerpts=excerpts,
                    verification=SegmentVerificationStats(
                        supported_claims=supported,
                        total_claims=total,
                        score=raw_score,
                        weighted_score=weighted,
                    ),
                )
            )

        # Build summary
        total_claims = len(all_claims_for_summary)
        supported_claims = sum(1 for c in all_claims_for_summary if c["is_supported"])
        if can_score:
            raw_rate = supported_claims / total_claims if total_claims > 0 else 0.0
            weighted_rate = self.scorer(all_claims_for_summary)
            fully_verified_segments = sum(
                1
                for so in segment_outputs
                if so.verification.weighted_score is not None
                and so.verification.weighted_score >= config.FULLY_VERIFIED_THRESHOLD
            )
            segment_rate = (
                fully_verified_segments / len(segment_outputs)
                if segment_outputs
                else 0.0
            )
        else:
            raw_rate = None
            weighted_rate = None
            segment_rate = None

        summary = VerificationSummary(
            total_segments=len(segment_outputs),
            total_claims=total_claims,
            supported_claims=supported_claims,
            raw_verification_rate=raw_rate,
            weighted_verification_rate=weighted_rate,
            segment_verification_rate=segment_rate,
        )

        return VerificationOutput(summary=summary, segments=segment_outputs)
