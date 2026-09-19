import re

from vetscore.models import (
    SourceExcerpt,
    SourceMetadata,
    TextSegment,
    VerificationInput,
)


class TextSegmentation:
    """Parses raw synthesis content into citation-linked segments."""
    def __call__(self, inp: VerificationInput) -> list[TextSegment]:
        return self.segment_with_plain_text(inp)[0]

    def segment_with_plain_text(self, inp: VerificationInput) -> tuple[list[TextSegment], str]:
        segments, plain_text = parse_with_plain_text(inp.synthesis_content)
        attach_source_metadata(segments, inp.sources)
        return segments, plain_text

    def get_plain_text(self, response: str) -> str:
        cleaned = preprocess(response)
        citations = find_all_citations(cleaned)
        return build_plain_text(cleaned, citations)


# === Text normalization ===


def normalize_segment_text(text: str) -> str:
    trimmed = text.strip()
    if not trimmed:
        return trimmed
    trimmed = re.sub(r"^[\s.,;:!?\-]+", "", trimmed)
    if not trimmed:
        return trimmed
    if trimmed.startswith("#"):
        return trimmed
    if not trimmed.endswith("."):
        trimmed += "."
    return trimmed


def normalize_heading_newlines(text: str) -> str:
    return re.sub(r"((?:^|\n)#{1,6}\s[^\n]*)\n(?!\n)", r"\1\n\n", text)


TABLE_ROW_RE = re.compile(r"\s*\|")


def normalize_tables(text: str) -> str:
    lines = text.split("\n")
    result = []
    i = 0
    while i < len(lines):
        if TABLE_ROW_RE.match(lines[i]):
            table_start = i
            while i < len(lines) and TABLE_ROW_RE.match(lines[i]):
                i += 1
            table_lines = lines[table_start:i]

            if len(table_lines) < 2:
                result.extend(table_lines)
                continue

            result.append("")
            for table_line in table_lines:
                line_stripped = table_line.strip()
                cells = [c.strip() for c in line_stripped.strip("|").split("|")]
                if all(re.match(r"^[-:\s]+$", c) for c in cells if c):
                    continue
                result.append(line_stripped)
                result.append("")
        else:
            result.append(lines[i])
            i += 1

    return "\n".join(result)


def get_base_source_id(source_id: str) -> str:
    return re.sub(r"[:#]L\d+(?:-\d+)?$", "", source_id)


def attach_source_metadata(
    segments: list[TextSegment], sources: dict[str, SourceMetadata]
) -> None:
    """Copy each source's provenance onto the excerpts that cite it.

    A citation id carries the line span it came from (``journal_abc:L47``);
    the metadata is per source, so the lookup drops that suffix.
    """
    if not sources:
        return
    for seg in segments:
        for src in seg.sources:
            meta = sources.get(src.source_id) or sources.get(
                get_base_source_id(src.source_id)
            )
            if meta is None:
                continue
            for field, value in meta.model_dump().items():
                setattr(src, field, value)


# === Parsing ===


def find_last_header_index(text: str) -> int:
    last = -1
    for m in re.finditer(r"(^|\n)(#+)\s", text):
        last = m.start()
    return last


LIST_ITEM_RE = re.compile(r"(?:^|\n)([ \t]*(?:[-*]|\d+[.)])[ \t])")


def trim_to_last_list_item(text: str) -> tuple[str, str]:
    matches = list(LIST_ITEM_RE.finditer(text))
    if not matches:
        return ("", text)
    last = matches[-1]
    split_pos = last.start()
    if split_pos < len(text) and text[split_pos] == "\n":
        split_pos += 1
    return (text[:split_pos], text[split_pos:])


SENTENCE_END_RE = re.compile(
    r'(?<!vs\.)(?<!approx\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)'  # skip abbreviations
    r'(?<=[.!?])"?\s+'  # period/!/? optionally followed by closing quote, then whitespace
    r'(?=[A-Z\*\d\("])'  # next sentence starts with uppercase, bold marker, digit, parenthesis, or quote
)


def trim_to_last_sentence(text: str) -> tuple[str, str]:
    matches = list(SENTENCE_END_RE.finditer(text))
    if not matches:
        return ("", text)
    last = matches[-1]
    return (text[: last.start() + 1], text[last.end() :])


def preprocess(text: str) -> str:
    cleaned = normalize_tables(text.strip())
    cleaned = normalize_heading_newlines(cleaned)
    return cleaned


# Citation regex: [text](source_id)
CITATION_RE = re.compile(
    r"\[([^\[\]]+(?:\[[^\]]*\][^\[\]]*)*)\]\s*\(([^()]+(?:\([^()]*\)[^()]*)*)\)"
)

# Fallback patterns for malformed citations
FALLBACK_PATTERNS = [
    # [**"text"**](source) — bold markers inside bracket
    re.compile(r"\[\*{1,2}\"([^\"]+)\"\*{0,2}\]\s*\(([^()]+)\)"),
    # ["text"][source] — square brackets for source
    re.compile(r'\["([^"]+)"\]\s*\[([^\[\]]+)\]'),
    # "text"](source) — missing opening [
    re.compile(r'(?<!\[)"([^"]{5,})"\]\s*\(([^()]+)\)'),
    # "text"(source) — missing both [ and ]; no space between
    # closing quote and ( to avoid false positives on regular
    # prose like "quoted phrase" (parenthetical note)
    re.compile(r'(?<!\[)"([^"]{5,})"\(([^()]+(?:\([^()]*\)[^()]*)*)\)'),
]


def strip_excerpt_wrapper(excerpt: str) -> str:
    if excerpt.startswith('"') and excerpt.endswith('"'):
        excerpt = excerpt[1:-1]
    excerpt = excerpt.strip("*")
    if excerpt.startswith('"') and excerpt.endswith('"'):
        excerpt = excerpt[1:-1]
    return excerpt


def find_all_citations(cleaned: str) -> list[dict]:
    citations: list[dict] = []
    covered: set[tuple[int, int]] = set()

    for m in CITATION_RE.finditer(cleaned):
        citations.append(
            {
                "source_id": m.group(2),
                "excerpt": strip_excerpt_wrapper(m.group(1)),
                "start": m.start(),
                "end": m.end(),
            }
        )
        covered.add((m.start(), m.end()))

    # Fallback patterns — skip matches overlapping with already-found citations
    for pat in FALLBACK_PATTERNS:
        for m in pat.finditer(cleaned):
            if any(
                not (m.end() <= cs or m.start() >= ce) for cs, ce in covered
            ):
                continue
            citations.append(
                {
                    "source_id": m.group(2),
                    "excerpt": m.group(1),
                    "start": m.start(),
                    "end": m.end(),
                }
            )
            covered.add((m.start(), m.end()))

    citations.sort(key=lambda c: c["start"])
    return citations


def build_plain_text(text: str, citations: list[dict]) -> str:
    """Remove citation markup from cleaned text, leaving only prose."""
    parts = []
    last_citation = 0
    for cit in citations:
        parts.append(text[last_citation:cit["start"]])
        last_citation = cit["end"]
    parts.append(text[last_citation:])
    return "".join(parts)


def assign_offsets(segments: list[TextSegment], plain_text: str) -> None:
    cursor = 0
    for seg in segments:
        search = seg.text
        if not search.startswith("#") and search.endswith("."):
            search = search[:-1]

        pos = plain_text.find(search, cursor)
        if pos < 0 and len(search) > 20:
            pos = plain_text.find(search[:20], cursor)
        if pos < 0:
            pos = cursor

        seg.start_offset = pos
        cursor = pos + max(len(search), 1)


def flush_segment(
    segments: list[TextSegment], text: str, sources: list[SourceExcerpt]
) -> list[SourceExcerpt]:
    """Normalize text, append it (with sources) as a segment if non-empty."""
    seg_text = normalize_segment_text(text)
    if seg_text or sources:
        segments.append(TextSegment(text=seg_text, sources=sources))
        return []
    return sources


def flush_paragraphs(
    segments: list[TextSegment], text: str, sources: list[SourceExcerpt]
) -> list[SourceExcerpt]:

    for chunk in re.split(r"\n\n+", text):
        sources = flush_segment(segments, chunk, sources)
    return sources


def should_close_segment(between_text: str) -> bool:
    if re.search(r"(^|\n)#+\s", between_text):
        return True
    if "\n\n" in between_text:
        return True
    if re.search(r"(^|\n)\s*(?:[-*]|\d+[.)])\s", between_text):
        return True
    return len(between_text.strip()) > 50


def parse_response_sources(response: str) -> list[TextSegment]:
    return parse_with_plain_text(response)[0]


def parse_with_plain_text(response: str) -> tuple[list[TextSegment], str]:
    """Segment response and return the citation-stripped text."""
    cleaned = preprocess(response)
    citations = find_all_citations(cleaned)
    plain_text = build_plain_text(cleaned, citations)

    if not citations:
        return [TextSegment(text=cleaned, sources=[], start_offset=0)], plain_text

    segments: list[TextSegment] = []
    current_text = ""
    current_sources: list[SourceExcerpt] = []
    last_idx = 0

    for i, citation in enumerate(citations):
        next_citation = citations[i + 1] if i + 1 < len(citations) else None
        preceding_text = cleaned[last_idx:citation["start"]]

        # Split at headers
        header_idx = find_last_header_index(preceding_text)
        if header_idx != -1:
            before_header = preceding_text[:header_idx]
            after_header = preceding_text[header_idx:]

            if current_text or before_header.strip():
                current_sources = flush_paragraphs(
                    segments, current_text + before_header, current_sources
                )
            current_text = after_header
        else:
            current_text += preceding_text

        # Trim at paragraph boundaries
        last_break = current_text.rfind("\n\n")
        if last_break != -1:
            before_break = current_text[:last_break]
            current_text = current_text[last_break:].lstrip("\n")
            current_sources = flush_paragraphs(segments, before_break, current_sources)

        # Trim uncited list items
        before_item, after_item = trim_to_last_list_item(current_text)
        if before_item and before_item != current_text:
            current_sources = flush_segment(segments, before_item, current_sources)
            current_text = after_item

        # Trim uncited leading sentences
        before_sent, last_sent = trim_to_last_sentence(current_text)
        if before_sent:
            current_sources = flush_segment(segments, before_sent, current_sources)
            current_text = last_sent

        current_sources.append(
            SourceExcerpt(source_id=citation["source_id"], excerpt=citation["excerpt"])
        )
        last_idx = citation["end"]

        should_close = next_citation is None or should_close_segment(
            cleaned[citation["end"] : next_citation["start"]]
        )
        if should_close:
            current_sources = flush_segment(segments, current_text, current_sources)
            current_text = ""

    # Handle remaining text after last citation
    if last_idx < len(cleaned):
        flush_paragraphs(segments, cleaned[last_idx:], [])

    # Merge empty-text segments into previous
    merged: list[TextSegment] = []
    for seg in segments:
        if seg.text == "" and seg.sources and merged:
            merged[-1].sources.extend(seg.sources)
        elif seg.text != "" or seg.sources:
            merged.append(seg)

    assign_offsets(merged, plain_text)

    return merged, plain_text
