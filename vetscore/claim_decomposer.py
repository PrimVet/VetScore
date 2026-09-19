import logging
import re

from vetscore import config
from vetscore.llm_backends import call_ollama_json, call_openrouter_json, infer_backend
from vetscore.models import TextSegment, parse_results_payload
from vetscore.prompts import buildDecompositionPrompt

logger = logging.getLogger(__name__)


class ClaimDecomposer:
    """Decomposes text segments into atomic claims."""

    def __init__(
        self,
        client,
        model: str,
        max_retries: int = 3,
        reasoning_level: str | None = None,
        temperature: float = config.DEFAULT_TEMPERATURE,
        provider: dict | None = None,
        schema_root: str = "array",
    ):
        self.client = client
        self.backend = infer_backend(client)
        self.model = model
        self.max_retries = max_retries
        self.reasoning_level = reasoning_level
        self.temperature = temperature
        self.provider = provider
        self.schema_root = schema_root

    def __call__(self, segments: list[TextSegment]) -> list[tuple[list[str], str]]:
        results = []

        for i, seg in enumerate(segments):
            previous = segments[:i]
            claims, ctx = self.decompose_segment(seg, previous, i)
            results.append((claims, ctx))

            total = sum(len(r[0]) for r in results)
            logger.info(
                "Decomposed segment %d/%d → %d claims (total: %d)",
                i + 1,
                len(segments),
                len(claims),
                total,
            )

        return results

    def decompose_segment(
        self,
        segment: TextSegment,
        previous_segments: list[TextSegment],
        segment_index: int,
        context_text: str | None = None,
    ) -> tuple[list[str], str]:
        if context_text is not None:
            context = self._get_previous_context(context_text)
        else:
            context = self._build_optimized_context(previous_segments, config.MAX_CONTEXT_CHARS)
        prompt = buildDecompositionPrompt(segment, context)
        context_label = f" for segment {segment_index}"

        if self.backend == "openrouter":
            array_schema = {"type": "array", "items": {"type": "string"}}
            if self.schema_root == "object":
                schema = {
                    "type": "object",
                    "properties": {"results": array_schema},
                    "required": ["results"],
                    "additionalProperties": False,
                }
            elif self.schema_root == "array":
                schema = array_schema
            else:
                raise ValueError(f"unknown schema_root {self.schema_root!r}")
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "atomic_claims", "strict": True, "schema": schema},
            }
            extra_body: dict = {
                "reasoning": config.OPENROUTER_REASONING[self.reasoning_level or "none"]
            }
            if self.provider:
                extra_body["provider"] = self.provider

            parsed, _usage = call_openrouter_json(
                self.client,
                model=self.model,
                prompt=prompt,
                response_format=response_format,
                extra_body=extra_body,
                temperature=self.temperature,
                max_retries=self.max_retries,
                unwrap=parse_results_payload,
                context_label=context_label,
            )

        elif self.backend == "ollama":
            schema = {"type": "array", "items": {"type": "string"}}
            if self.schema_root == "object":
                schema = {
                    "type": "object",
                    "properties": {"results": schema},
                    "required": ["results"],
                    "additionalProperties": False,
                }
            elif self.schema_root != "array":
                raise ValueError(f"unknown schema_root {self.schema_root!r}")

            parsed, _usage = call_ollama_json(
                self.client,
                model=self.model,
                prompt=prompt,
                response_format=schema,
                think=config.OLLAMA_THINK[self.reasoning_level or "none"],
                temperature=self.temperature,
                max_retries=self.max_retries,
                unwrap=parse_results_payload,
                context_label=context_label,
            )

        else:
            raise ValueError(f"unknown backend {self.backend!r}")

        if parsed is None:
            return [], context
        return [str(c) for c in parsed], context

    @staticmethod
    def _get_previous_context(context_text: str | None) -> str:
        if not context_text:
            return "(No previous context)"
        if len(context_text) > config.MAX_CONTEXT_CHARS:
            context = context_text[-config.MAX_CONTEXT_CHARS:]
            # Drop the leading partial sentence/paragraph
            first_para = context.find("\n\n")
            if first_para != -1 and first_para < 200:
                context = context[first_para + 2:]
            else:
                m = re.search(r"[.!?]\s+", context)
                if m and m.end() < 200:
                    context = context[m.end():]
            return context
        return context_text

    @classmethod
    def _build_optimized_context(
        cls, previous_segments: list[TextSegment], max_chars: int = 2000
    ) -> str:
        if not previous_segments:
            return "(No previous context)"

        # Collect texts that fit within the char budget (most-recent first)
        selected: list[str] = []
        total_chars = 0
        for i in range(len(previous_segments) - 1, -1, -1):
            seg_text = previous_segments[i].text
            if total_chars + len(seg_text) > max_chars and selected:
                break
            selected.insert(0, seg_text)
            total_chars += len(seg_text)

        top_level = cls._detect_top_level_heading(selected)

        # Trim to content from the last top-level header onward
        if top_level:
            header_re = re.compile(r"(?:^|\n)(#{" + str(top_level) + r"})\s")
            last_header_seg = -1
            last_header_pos = -1
            for idx, text in enumerate(selected):
                for m in header_re.finditer(text):
                    last_header_seg = idx
                    last_header_pos = m.start()

            if last_header_seg >= 0:
                # Keep from the last top-level header onward
                trimmed_first = selected[last_header_seg][last_header_pos:].lstrip("\n")
                selected = [trimmed_first] + selected[last_header_seg + 1:]

        return "\n\n".join(selected)

    @staticmethod
    def _detect_top_level_heading(texts: list[str], min_level: int = 7) -> int:
        top_level = min_level
        for t in texts:
            for m in re.finditer(r"(?:^|\n)(#{1,6})\s", t):
                top_level = min(top_level, len(m.group(1)))
        return top_level if top_level < min_level else 0
