import logging

from pydantic import BaseModel

from vetscore.models import SourceExcerpt, TextSegment

logger = logging.getLogger(__name__)


class DatasetExcerpt(BaseModel):
    excerpt: str | None = None
    source_id: str = ""
    source_url: str | None = None
    source_title: str | None = None
    source_authors: list[str] = []
    source_journal: str | None = None
    source_year: int | None = None
    license: str | None = None
    match: str | None = None
    verbatim: bool | None = None


class SegmentRecord(BaseModel):
    id: str | None = None
    query_id: int | None = None
    model: str | None = None
    query: str | None = None
    topic: str | None = None
    segment_index: int | None = None
    text: str
    excerpts: list[DatasetExcerpt] = []
    claims: list[str] = []


def looks_like_segment_dataset(raw) -> bool:
    if not isinstance(raw, dict):
        return False
    segments = raw.get("segments")
    if not isinstance(segments, list) or not segments:
        return False
    first = segments[0]
    if not isinstance(first, dict) or "text" not in first:
        return False
    return any(k in first for k in ("claims", "excerpts"))


def parse_segment_dataset(raw) -> list[SegmentRecord]:
    return [SegmentRecord.model_validate(s) for s in raw["segments"]]


def to_text_segments(records: list[SegmentRecord]) -> list[TextSegment]:
    segments: list[TextSegment] = []
    blanked = 0
    uncited: list[str] = []
    for rec in records:
        sources = [
            SourceExcerpt(**e.model_dump())
            for e in rec.excerpts
            if e.excerpt
        ]
        blanked += len(rec.excerpts) - len(sources)
        if not sources:
            uncited.append(rec.id or f"index {len(segments)}")
        segments.append(TextSegment(text=rec.text, sources=sources))

    if blanked:
        logger.warning(
            "Dropped %d excerpt(s) with no text (blanked in the public release "
            "for licensing reasons).", blanked
        )
    if uncited:
        logger.warning(
            "%d record(s) have no excerpt text left and will be skipped "
            "(no claims verified, no scores): %s%s",
            len(uncited),
            ", ".join(uncited[:5]),
            ", ..." if len(uncited) > 5 else "",
        )
    return segments


def to_claims(records: list[SegmentRecord]) -> list[list[str]]:
    return [rec.claims for rec in records]


def to_questions(records: list[SegmentRecord]) -> list[str | None]:
    return [rec.query for rec in records]
