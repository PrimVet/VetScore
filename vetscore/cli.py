"""CLI entry point: vetscore run_config.yaml"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

from dotenv import find_dotenv, load_dotenv

from vetscore.claim_decomposer import ClaimDecomposer
from vetscore.harm_potential import HarmPotentialScorer
from vetscore.models import SourceExcerpt, TextSegment, VerificationInput
from vetscore.pipeline import Pipeline
from vetscore.run_config import (
    DEFAULT_OPENROUTER_BASE_URL,
    ConfigError,
    RunConfig,
    load_run_config,
)
from vetscore.segment_dataset import (
    looks_like_segment_dataset,
    parse_segment_dataset,
    to_claims,
    to_questions,
    to_text_segments,
)
from vetscore.text_segmentation import TextSegmentation
from vetscore.verifier import Verifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_config",
        help="YAML run config (e.g. configs/gemma-4-31b.yaml).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    load_dotenv()

    try:
        cfg = load_run_config(args.run_config)
    except ConfigError as exc:
        exit_with_error(str(exc))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    client, component_kwargs = build_backend(cfg)
    input_path = Path(cfg.input)

    if input_path.is_dir():
        exit_with_error(f"Input must be a file, not a directory: {input_path}")
    if not input_path.is_file():
        exit_with_error(f"Input not found: {input_path}")

    process_file(input_path, cfg.output, client, cfg, component_kwargs)


def exit_with_error(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    sys.exit(1)


def build_backend(cfg: RunConfig) -> tuple[object, dict]:
    kwargs = {
        "temperature": cfg.temperature,
        "reasoning_level": cfg.reasoning_level,
        "max_retries": cfg.max_retries,
        "schema_root": cfg.schema_root,
        "num_samples": cfg.num_samples,
    }

    if cfg.backend == "ollama":
        try:
            from ollama import Client
        except ImportError:
            exit_with_error(
                "The ollama backend needs the ollama package, which "
                "is an optional extra: pip install -e '.[ollama]')."
            )
        return Client(host=cfg.base_url), kwargs

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        exit_with_error("Set OPENROUTER_API_KEY environment variable.")
    if cfg.provider is not None:
        kwargs["provider"] = cfg.provider

    try:
        from openai import OpenAI
    except ImportError:
        exit_with_error(
            "The openrouter backend needs the 'openai' package (>=1.0), "
            "normally installed with vetscore itself: pip install 'openai>=1.0'"
        )

    client = OpenAI(base_url=cfg.base_url or DEFAULT_OPENROUTER_BASE_URL, api_key=api_key)
    return client, kwargs


def decomposer_kwargs(component_kwargs: dict) -> dict:
    return {k: v for k, v in component_kwargs.items() if k != "num_samples"}


def process_segment_dataset(
    raw: dict,
    output_path: str | None,
    client,
    cfg: RunConfig,
    component_kwargs: dict,
) -> None:
    records = parse_segment_dataset(raw)
    print(f"Loaded {len(records)} segment records", file=sys.stderr)

    if not cfg.skip_segmentation:
        exit_with_error(
            "This input is a segment dataset — its records are already one "
            "segment each, so there is nothing to segment. Set "
            "skip_segmentation: true in the run config."
        )

    if cfg.segment_indices_from:
        ref_path = Path(cfg.segment_indices_from)
        if not ref_path.is_file():
            exit_with_error(f"segment_indices_from file not found: {ref_path}")
        with open(ref_path) as f:
            ref_data = json.load(f)
        ref_ids = {
            seg["id"]
            for seg in ref_data.get("segments", [])
            if seg.get("id") and seg.get("atomic_claims")
        }
        if not ref_ids:
            exit_with_error(
                f"{ref_path} has no segments with an 'id' and atomic claims — "
                "it does not look like the output of a segment-dataset run."
            )
        records = [r for r in records if r.id in ref_ids]
        print(
            f"Restricted to {len(records)} records from {ref_path.name}",
            file=sys.stderr,
        )

    if cfg.sample_segments is not None and cfg.sample_segments < len(records):
        import random

        rng = random.Random(cfg.sample_seed)
        picked = sorted(rng.sample(range(len(records)), cfg.sample_segments))
        records = [records[i] for i in picked]
        print(f"Sampled {len(records)} records", file=sys.stderr)

    if not records:
        exit_with_error("No segment records left to process.")

    if cfg.skip_decomposition and not any(r.claims for r in records):
        exit_with_error(
            "skip_decomposition is set, but no record carries claims, so "
            "every segment would be verified against an empty claim list. "
            "Drop skip_decomposition to decompose these records instead."
        )

    pipeline = Pipeline(
        text_segmentation=None,
        decomposer=None if cfg.skip_decomposition else ClaimDecomposer(
            client, model=cfg.model, **decomposer_kwargs(component_kwargs)
        ),
        harm_scorer=None if cfg.skip_scoring else HarmPotentialScorer(
            client, model=cfg.model, **component_kwargs
        ),
        verifier=None if cfg.skip_verification else Verifier(
            client, model=cfg.model, **component_kwargs
        ),
    )
    def checkpoint(partial) -> None:
        attach_record_identity(partial, records)
        write_output(partial.model_dump(exclude_none=True), output_path, quiet=True)
        print(
            f"  checkpoint: {len(partial.segments)} segments -> {output_path}",
            file=sys.stderr,
        )

    result = pipeline(
        VerificationInput(synthesisContent=""),
        precomputed_segments=to_text_segments(records),
        precomputed_claims=to_claims(records) if cfg.skip_decomposition else None,
        independent_segments=True,
        user_questions=to_questions(records),
        checkpoint_every=cfg.checkpoint_every,
        on_checkpoint=checkpoint if cfg.checkpoint_every else None,
    )
    attach_record_identity(result, records)
    write_output(result.model_dump(exclude_none=True), output_path)


def attach_record_identity(result, records) -> None:
    for out_seg, rec in zip(result.segments, records):
        out_seg.id = rec.id
        out_seg.query_id = rec.query_id
        out_seg.model = rec.model
        out_seg.query = rec.query
        out_seg.topic = rec.topic
        out_seg.source_segment_index = rec.segment_index


def write_output(output_json: dict, output_path: str | None, quiet: bool = False) -> None:
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(output_json, f, indent=2)
            os.replace(tmp, out)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        if not quiet:
            print(f"Output written to {output_path}", file=sys.stderr)
    else:
        json.dump(output_json, sys.stdout, indent=2)
        print(file=sys.stdout)


def load_input(path: Path):
    if path.suffix == ".jsonl":
        records = []
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                exit_with_error(f"{path.name} line {lineno} is not valid JSON: {exc}")
        if not records:
            exit_with_error(f"{path.name} has no records.")
        return {"segments": records}

    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            exit_with_error(f"{path.name} is not valid JSON: {exc}")


def process_file(
    input_path: Path,
    output_path: str | None,
    client,
    cfg: RunConfig,
    component_kwargs: dict,
) -> None:
    raw = load_input(input_path)

    if looks_like_segment_dataset(raw):
        process_segment_dataset(raw, output_path, client, cfg, component_kwargs)
        return

    fixed_segment_indices = None
    if cfg.segment_indices_from:
        ref_path = Path(cfg.segment_indices_from)
        if ref_path.is_file():
            with open(ref_path) as f:
                ref_data = json.load(f)
            fixed_segment_indices = [
                s["segment_index"] for s in ref_data.get("segments", [])
                if s.get("atomic_claims")
            ]
            if cfg.sample_segments is not None and len(fixed_segment_indices) > cfg.sample_segments:
                fixed_segment_indices = fixed_segment_indices[:cfg.sample_segments]

    precomputed_segments = None
    precomputed_sampled_indices = None
    precomputed_plain_text = None
    if cfg.skip_segmentation:
        if "textSegments" not in raw or "sampledSegmentIndices" not in raw:
            exit_with_error(
                f"skip_segmentation is set, but {input_path.name} carries no "
                "'textSegments'/'sampledSegmentIndices' to use instead. Drop "
                "skip_segmentation, or point `input` at a pre-segmented file."
            )
        precomputed_segments = [
            TextSegment(
                text=s["text"],
                sources=[
                    SourceExcerpt(source_id=src["sourceId"], excerpt=src["excerpt"])
                    for src in s.get("sources", [])
                ],
                start_offset=s.get("startOffset"),
            )
            for s in raw["textSegments"]
        ]
        precomputed_sampled_indices = raw["sampledSegmentIndices"]
        precomputed_plain_text = raw.get("plainText")

    if cfg.skip_decomposition:
        exit_with_error(
            f"skip_decomposition is set, but {input_path.name} carries no "
            "atomic claims — only a segment dataset does. Drop "
            "skip_decomposition to decompose this input."
        )

    inp = VerificationInput.model_validate(raw)
    pipeline = Pipeline(
        text_segmentation=TextSegmentation(),
        decomposer=ClaimDecomposer(
            client, model=cfg.model, **decomposer_kwargs(component_kwargs)
        ),
        harm_scorer=None if cfg.skip_scoring else HarmPotentialScorer(
            client, model=cfg.model, **component_kwargs
        ),
        verifier=None if cfg.skip_verification else Verifier(
            client, model=cfg.model, **component_kwargs
        ),
    )
    def checkpoint(partial) -> None:
        write_output(partial.model_dump(exclude_none=True), output_path, quiet=True)
        print(
            f"  checkpoint: {len(partial.segments)} segments -> {output_path}",
            file=sys.stderr,
        )

    result = pipeline(
        inp,
        sample_segments=cfg.sample_segments,
        fixed_segment_indices=fixed_segment_indices,
        sample_seed=cfg.sample_seed,
        precomputed_segments=precomputed_segments,
        precomputed_sampled_indices=precomputed_sampled_indices,
        plain_text=precomputed_plain_text,
        checkpoint_every=cfg.checkpoint_every,
        on_checkpoint=checkpoint if cfg.checkpoint_every else None,
    )
    write_output(result.model_dump(exclude_none=True), output_path)


if __name__ == "__main__":
    main()
