import json
import logging
import re
from typing import Any, Callable

from vetscore.models import TokenUsage

logger = logging.getLogger(__name__)

_NON_RETRYABLE_STATUS = frozenset({400, 404})


def infer_backend(client) -> str:
    if type(client).__module__.split(".")[:1] == ["ollama"]:
        return "ollama"
    return "openrouter"


def call_openrouter_json(
    client,
    *,
    model: str,
    prompt: str,
    response_format: dict,
    extra_body: dict,
    temperature: float,
    max_retries: int,
    max_output_tokens: int | None = None,
    unwrap: Callable[[Any], Any] = lambda parsed: parsed,
    context_label: str = "",
) -> tuple[Any | None, TokenUsage]:
    request_kwargs: dict = {}
    if max_output_tokens is not None:
        request_kwargs["max_tokens"] = max_output_tokens

    total_usage = TokenUsage()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format=response_format,
                extra_body=extra_body,
                **request_kwargs,
            )
            usage_meta = response.usage
            details = getattr(usage_meta, "completion_tokens_details", None)
            thinking_tokens = getattr(details, "reasoning_tokens", 0) or 0
            total_usage = total_usage + TokenUsage(
                input_tokens=getattr(usage_meta, "prompt_tokens", 0) or 0,
                output_tokens=(getattr(usage_meta, "completion_tokens", 0) or 0) - thinking_tokens,
                thinking_tokens=thinking_tokens,
            )

            text = response.choices[0].message.content or ""
            if not text:
                logger.warning(
                    "Empty OpenRouter response%s (attempt %d/%d)",
                    context_label,
                    attempt + 1,
                    max_retries,
                )
                continue

            parsed = unwrap(json.loads(text))
            if not isinstance(parsed, list):
                logger.warning(
                    "Non-array OpenRouter response%s (attempt %d/%d)",
                    context_label,
                    attempt + 1,
                    max_retries,
                )
                continue

            return parsed, total_usage

        except Exception as e:
            logger.error(
                "OpenRouter call failed%s (attempt %d/%d): %s",
                context_label,
                attempt + 1,
                max_retries,
                e,
            )

    logger.error("All %d OpenRouter attempts failed%s", max_retries, context_label)
    return None, total_usage


def call_ollama_json(
    client,
    *,
    model: str,
    prompt: str,
    response_format: dict,
    temperature: float,
    max_retries: int,
    think: bool | str | None = None,
    max_output_tokens: int | None = None,
    unwrap: Callable[[Any], Any] = lambda parsed: parsed,
    context_label: str = "",
) -> tuple[Any | None, TokenUsage]:
    options: dict = {"temperature": temperature}
    if max_output_tokens is not None:
        options["num_predict"] = max_output_tokens

    total_usage = TokenUsage()
    for attempt in range(max_retries):
        try:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format=response_format,
                think=think,
                options=options,
            )
            total_usage = total_usage + TokenUsage(
                input_tokens=getattr(response, "prompt_eval_count", 0) or 0,
                output_tokens=getattr(response, "eval_count", 0) or 0,
            )

            text = response.message.content or ""
            if not text:
                logger.warning(
                    "Empty Ollama response%s (attempt %d/%d)",
                    context_label,
                    attempt + 1,
                    max_retries,
                )
                continue

            parsed = unwrap(json.loads(_strip_think_block(text)))
            if not isinstance(parsed, list):
                logger.warning(
                    "Non-array Ollama response%s (attempt %d/%d)",
                    context_label,
                    attempt + 1,
                    max_retries,
                )
                continue

            return parsed, total_usage

        except Exception as e:
            logger.error(
                "Ollama call failed%s (attempt %d/%d): %s",
                context_label,
                attempt + 1,
                max_retries,
                e,
            )
            if getattr(e, "status_code", None) in _NON_RETRYABLE_STATUS:
                logger.error("Not retrying%s: the request was rejected", context_label)
                return None, total_usage

    logger.error("All %d Ollama attempts failed%s", max_retries, context_label)
    return None, total_usage


def _strip_think_block(text: str) -> str:
    # A closing tag with no opener before it: the template pre-injected the
    # opening tag, so everything up to the closing tag is thinking.
    head, closing_tag, tail = text.partition("</think>")
    if closing_tag and "<think>" not in head:
        text = tail

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # An unterminated block: everything from it onward is thinking.
    text = re.split(r"<think>", text, maxsplit=1)[0]
    return text.strip()
