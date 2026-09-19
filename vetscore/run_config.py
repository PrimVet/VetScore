from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vetscore import config

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ConfigError(Exception):
    pass


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: str = Field(
        description="Input file: a JSONL segment dataset, or a JSON synthesis to segment."
    )
    output: str | None = Field(default=None, description="Output JSON file.")
    backend: Literal["openrouter", "ollama"] = "openrouter"
    model: str = Field(description="Model id in the backend's own namespace.")
    base_url: str | None = Field(
        default=None,
        description=(
            f"OpenAI-compatible endpoint for openrouter (default: "
            f"{DEFAULT_OPENROUTER_BASE_URL}) or Ollama host for ollama "
            "(default: the OLLAMA_HOST env var, else http://localhost:11434)."
        ),
    )
    provider: dict[str, Any] | None = Field(
        default=None,
        description=(
            'openrouter-only: provider-routing object, e.g. {"only": '
            '["open-inference"]}.'
        ),
    )
    temperature: float = config.DEFAULT_TEMPERATURE
    reasoning_level: str = Field(
        default="none",
        description=(
            "Reasoning effort. OpenRouter takes a key of "
            "config.OPENROUTER_REASONING, ollama one of config.OLLAMA_THINK."
        ),
    )
    max_retries: int = Field(default=3, ge=1)
    schema_root: Literal["array", "object"] = Field(
        default="array",
        description=(
            "JSON response shape: 'array' (root list or 'object' "
            "(list wrapped under 'results', required by OpenAI "
            "models, which reject a root-level array)."
        ),
    )
    num_samples: int = Field(
        default=1,
        ge=1,
        description="LLM samples per segment for scoring and verification.",
    )
    skip_segmentation: bool = Field(
        default=False,
        description=(
            "Use the segments the input already has instead of splitting "
            "the synthesis text."
        ),
    )
    skip_decomposition: bool = Field(
        default=False,
        description=(
            "Use the atomic claims the input already has instead of asking "
            "the model for them."
        ),
    )
    skip_scoring: bool = False
    skip_verification: bool = False
    checkpoint_every: int | None = Field(
        default=None,
        ge=1,
        description="Rewrite `output` every N processed segments.",
    )
    sample_segments: int | None = Field(
        default=None, ge=1, description="Randomly sample N segments (default: all)."
    )
    sample_seed: int | None = Field(
        default=None, description="Seed for `sample_segments`, for reproducibility."
    )
    segment_indices_from: str | None = Field(
        default=None,
        description=(
            "Restrict the run to the segments of an existing output file, so "
            "two models can be compared on exactly the same ones."
        ),
    )

    @model_validator(mode="after")
    def _check_run_is_possible(self) -> "RunConfig":
        if self.skip_scoring and self.skip_verification:
            raise ValueError(
                "skip_scoring and skip_verification are both set — nothing "
                "would be left to run."
            )
        if self.skip_decomposition and not self.skip_segmentation:
            raise ValueError(
                "skip_decomposition requires skip_segmentation"
            )
        if self.num_samples > 1 and self.temperature == 0:
            raise ValueError(
                f"num_samples is {self.num_samples} but temperature is 0, so "
                "every sample would be identical - raise the temperature or "
                "decrease num_samples to 1."
            )
        if self.checkpoint_every is not None and not self.output:
            raise ValueError("checkpoint_every needs an output file.")

        levels = (
            config.OPENROUTER_REASONING
            if self.backend == "openrouter"
            else config.OLLAMA_THINK
        )
        if self.reasoning_level not in levels:
            raise ValueError(
                f"reasoning_level {self.reasoning_level!r} is not one of "
                f"{', '.join(levels)} (the {self.backend} vocabulary)."
            )
        if self.backend == "ollama" and self.provider is not None:
            raise ValueError("provider only applies to the openrouter backend.")
        return self


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"Run config not found: {path}") from None
    except OSError as exc:
        raise ConfigError(f"Cannot read run config {path}: {exc}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from None

    if raw is None:
        raise ConfigError(f"{path} is empty.")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} must hold a mapping of settings, not {type(raw).__name__}."
        )

    try:
        return RunConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"{path} is not a valid run config:\n{exc}") from None
