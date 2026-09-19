from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    source_url: str | None = None
    source_title: str | None = None
    source_authors: list[str] = []
    source_journal: str | None = None
    source_year: int | None = None
    license: str | None = None


class VerificationInput(BaseModel):
    synthesis_content: str = Field(alias="synthesisContent")
    user_question: str | None = Field(default=None, alias="userQuestion")
    sources: dict[str, SourceMetadata] = {}

    model_config = {"populate_by_name": True}


class SourceExcerpt(BaseModel):
    source_id: str
    excerpt: str
    source_url: str | None = None
    source_title: str | None = None
    source_authors: list[str] = []
    source_journal: str | None = None
    source_year: int | None = None
    license: str | None = None
    match: str | None = None
    verbatim: bool | None = None


class TextSegment(BaseModel):
    text: str
    sources: list[SourceExcerpt] = []
    start_offset: int | None = None


def injected_reason(text: str) -> str:
    return f"[{text}]"


NOT_RETURNED = injected_reason("Not returned by LLM")
NO_SOURCES = injected_reason("No direct sources available")
INVALID_ITEM = injected_reason("Invalid item in response")
DEFAULT_ON_ERROR = injected_reason("Default score due to error")


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
        )


def list_result_schema(item_properties: dict, schema_root: str = "array") -> dict:
    item_schema = {
        "type": "object",
        "properties": item_properties,
        "required": list(item_properties),
        "additionalProperties": False,
    }
    if schema_root == "array":
        return {"type": "array", "items": item_schema}
    if schema_root == "object":
        return {
            "type": "object",
            "properties": {"results": {"type": "array", "items": item_schema}},
            "required": ["results"],
            "additionalProperties": False,
        }
    raise ValueError(f"unknown schema_root {schema_root!r}")


def openrouter_response_format(
    name: str, item_properties: dict, schema_root: str = "array"
) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": list_result_schema(item_properties, schema_root),
        },
    }


def parse_results_payload(parsed):
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return parsed.get("results")
    return None


class AtomicClaimOutput(BaseModel):
    text: str
    harm_potential: int | float | None = None
    harm_potential_samples: list[int] | None = None
    harm_potential_reasoning: str | None = None
    harm_potential_reasoning_samples: list[str] | None = None
    is_supported: bool | float | None = None
    is_supported_samples: list[bool] | None = None
    is_supported_reasoning: str
    is_supported_reasoning_samples: list[str] | None = None


class SegmentVerificationStats(BaseModel):
    supported_claims: int
    total_claims: int
    score: float | None = None
    weighted_score: float | None = None


class SegmentOutput(BaseModel):
    segment_index: int
    id: str | None = None
    query_id: int | None = None
    model: str | None = None
    query: str | None = None
    topic: str | None = None
    source_segment_index: int | None = None

    text: str
    decomposition_context: str | None = None
    atomic_claims: list[AtomicClaimOutput] = []
    excerpts: list[dict] = []
    verification: SegmentVerificationStats


class VerificationSummary(BaseModel):
    total_segments: int
    total_claims: int
    supported_claims: int
    raw_verification_rate: float | None = None
    weighted_verification_rate: float | None = None
    segment_verification_rate: float | None = None


class VerificationOutput(BaseModel):
    pipeline: str = "paraphrase-verification"
    summary: VerificationSummary
    segments: list[SegmentOutput] = []
