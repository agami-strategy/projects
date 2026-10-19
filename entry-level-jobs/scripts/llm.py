"""DeepSeek provider layer shared by the labelling scripts.

DeepSeek exposes an OpenAI-compatible endpoint (https://api.deepseek.com). This module wraps
the one call pattern every labelling script needs: system prompt + user message in, a
validated Pydantic object out.

DeepSeek has a JSON mode (response_format json_object) but no schema enforcement, so the
schema is placed in the prompt and the reply is validated locally; an invalid or empty reply
is retried with the validation error fed back. Prefix caching is automatic on DeepSeek's side
and is reported in usage as prompt_cache_hit_tokens / prompt_cache_miss_tokens, so the large
OSCA list in a repeated system prompt is cheap without any cache markers.

Needs DEEPSEEK_API_KEY in the environment or in <project>/.env.
"""
# %% Imports
# Standard library plus two third-party packages: `openai` (used purely as an HTTP client for the
# OpenAI-compatible DeepSeek endpoint; no OpenAI models are ever called) and `pydantic` (schema
# definition + local validation of the model's JSON reply). Nothing here touches the network at
# import time; the first API client is built lazily in client().
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import TypeVar

import openai
from pydantic import BaseModel, ValidationError

# The scripts are run as `python scripts/<name>.py`, so the scripts folder is not on sys.path by
# default. This insert makes `from common import ...` work regardless of the working directory.
# load_env is what reads DEEPSEEK_API_KEY from <project>/.env.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_env  # noqa: E402

# %% Provider constants and price table
# BASE_URL points the OpenAI SDK at DeepSeek. Everything downstream (JSON mode, usage fields,
# the `thinking` extra_body key) is DeepSeek-specific behaviour on top of that compatible API.
BASE_URL = "https://api.deepseek.com"

# USD per million tokens at PEAK rates (01:00-04:00 and 06:00-10:00 UTC, Mon-Fri). Off-peak is
# half of these. Cost summaries printed by the scripts use peak rates, so they are upper bounds.
# MODELS maps a short CLI-friendly key ("flash", "pro") to the provider model id and its price
# per token class. The three price keys mirror what DeepSeek reports in `usage`: cached prompt
# tokens (cache_hit), uncached prompt tokens (cache_miss) and completion tokens (output).
# Reviewer check: prices are hard-coded and will drift; update them here if DeepSeek changes
# its list price, and note that cost() / estimate_cost() below are only as accurate as this table.
MODELS = {
    "flash": {"id": "deepseek-flash",  "price": {"cache_hit": 0.006, "cache_miss": 0.30, "output": 1.20}},
    "pro":   {"id": "deepseek-v4-pro", "price": {"cache_hit": 0.044, "cache_miss": 1.32, "output": 3.96}},
}
# "flash" is the default everywhere because the labelling tasks are cheap classification jobs;
# scripts expose --model pro for a stronger model at ~4x the price.
DEFAULT_MODEL = "flash"
# Canonical order of the usage dict keys; scripts sum usage across ads using this tuple.
USAGE_KEYS = ("cache_hit", "cache_miss", "output")

# Generic type variable so complete_json() returns the concrete schema class the caller passed in
# (e.g. Label or Seniority) rather than a bare BaseModel; purely for type checkers.
T = TypeVar("T", bound=BaseModel)


# %% Refusal exception
# Raised by complete_json() only after every retry has failed. Callers (label_one in each script)
# catch it and record {"id": ..., "error": ...} for that ad instead of aborting the whole run.
# The message carries the last failure reason (empty reply, cut off, JSON/validation error).
class Refusal(Exception):
    """Raised when the model returns nothing usable after all retries."""


# %% Lazy singleton API client
# One openai.OpenAI instance is shared by all worker threads (the SDK client is thread-safe).
# It is built on first use rather than at import so that --dry-run and --review-only paths, which
# never call the API, work without an API key present.
_client: openai.OpenAI | None = None


def client() -> openai.OpenAI:
    global _client
    if _client is None:
        # load_env() reads <project>/.env and, by design, lets the file override an existing shell
        # variable (see common.py for why). The key must exist after that or we exit with a hint.
        load_env()
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            sys.exit("DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add your key.")
        # max_retries=4 is the SDK's own transport-level retry (connection errors, 429, 5xx) with
        # exponential backoff. It is separate from the content-level retry loop in complete_json().
        # timeout=600s is generous because thinking mode with the ~43k-token OSCA prompt can be slow.
        _client = openai.OpenAI(api_key=key, base_url=BASE_URL, max_retries=4, timeout=600)
    return _client


# %% Schema instruction appended to the system prompt
# DeepSeek's JSON mode guarantees syntactically valid JSON but not a particular shape, so the
# Pydantic JSON schema is serialised into the prompt as the contract. This text is appended to the
# END of the system prompt so that the large, stable part (e.g. the OSCA list) stays a byte-identical
# prefix across calls and keeps hitting the provider's prefix cache.
# ensure_ascii=False keeps any non-ASCII characters in field descriptions readable to the model.
# Design choice: if a provider with native structured outputs were used instead, this function and
# the local validation in complete_json() would become redundant.
def schema_instruction(schema: type[BaseModel]) -> str:
    """Text appended to the system prompt so JSON mode knows the shape to produce."""
    return (
        "\n\nRespond with a single JSON object and nothing else. It must validate against this JSON schema:\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        + "\nUse exactly the field names shown. Do not wrap the object in markdown."
    )


# %% complete_json: the single call pattern used by every labelling script
# Inputs: system prompt text, user message text, a Pydantic schema class, model key, token cap,
# thinking flag, and number of content-level retries.
# Output: (validated schema instance, usage dict summed over all attempts, provider model id).
# Raises Refusal if no attempt produced a valid object.
#
# Retry design: DeepSeek occasionally returns empty content in JSON mode, and without schema
# enforcement it can return JSON that parses but fails Pydantic validation (wrong field name, wrong
# type, invalid Literal). Each failure is fed back as a short assistant/user exchange so the model
# can correct itself. Retries reuse the same system prompt, so they are still prefix-cache hits.
# Reviewer could tweak: `retries` (default 2 = up to 3 attempts), the 1s sleep between attempts,
# or `max_tokens` (a too-small cap shows up as finish_reason == "length").
def complete_json(
    system: str,
    user: str,
    schema: type[T],
    model_key: str = DEFAULT_MODEL,
    max_tokens: int = 4000,
    thinking: bool = False,
    retries: int = 2,
) -> tuple[T, dict[str, int], str]:
    """Return (parsed object, usage {cache_hit, cache_miss, output}, model id used)."""
    model = MODELS[model_key]
    # The schema text is appended to the caller's system prompt here, once per call.
    messages = [
        {"role": "system", "content": system + schema_instruction(schema)},
        {"role": "user", "content": user},
    ]
    # Usage is accumulated across attempts so a retried ad is charged for every attempt it caused.
    usage_total = {k: 0 for k in USAGE_KEYS}
    last_error = ""
    for attempt in range(retries + 1):
        # response_format json_object switches on DeepSeek JSON mode. The `thinking` key is a
        # DeepSeek extension that the OpenAI SDK does not know about, hence extra_body. Thinking is
        # off by default: it is slower and produces more (billed) output tokens.
        resp = client().chat.completions.create(
            model=model["id"],
            messages=messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        )
        u = resp.usage
        # DeepSeek adds prompt_cache_hit_tokens / prompt_cache_miss_tokens to the standard usage
        # block. getattr with defaults keeps this robust if the fields are absent; the cache_miss
        # fallback derives it as prompt_tokens minus hits so the two always sum to prompt_tokens.
        usage_total["cache_hit"] += getattr(u, "prompt_cache_hit_tokens", 0) or 0
        usage_total["cache_miss"] += getattr(u, "prompt_cache_miss_tokens", None) or (u.prompt_tokens - (getattr(u, "prompt_cache_hit_tokens", 0) or 0))
        usage_total["output"] += u.completion_tokens or 0
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        # Three failure classes, checked in order: truncated by max_tokens, empty content (a known
        # DeepSeek JSON-mode quirk), or content that is not valid JSON / does not fit the schema.
        if choice.finish_reason == "length":
            last_error = "response was cut off by max_tokens"
        elif not content:
            last_error = "empty response"  # documented DeepSeek JSON-mode failure mode
        else:
            try:
                # Success path: json.loads then Pydantic validation (types, Literal values, required
                # fields). resp.model is the provider's canonical model id, stored with each label.
                return schema.model_validate(json.loads(content)), usage_total, resp.model
            except (json.JSONDecodeError, ValidationError) as e:
                # Validation errors can be long (one entry per bad field); truncate for the prompt.
                last_error = f"{type(e).__name__}: {str(e)[:500]}"
        if attempt < retries:
            # feed the failure back so the retry can correct it
            # messages[:2] keeps only the original system + user turns, so the conversation never
            # grows beyond four turns no matter how many retries happen. The bad reply is echoed as
            # the assistant turn ("(empty)" if there was none) and the error becomes the new user turn.
            messages = messages[:2] + [
                {"role": "assistant", "content": content or "(empty)"},
                {"role": "user", "content": f"That response was invalid ({last_error}). Return only the JSON object, complete and valid."},
            ]
            time.sleep(1)
    raise Refusal(last_error)


# %% Cost helpers
# cost(): actual spend for a usage dict (as returned by complete_json or summed over a run) at the
# peak list price of the given model. Divide by two for an off-peak run.
def cost(usage: dict[str, int], model_key: str) -> float:
    price = MODELS[model_key]["price"]
    return sum(usage.get(k, 0) * price[k] / 1e6 for k in USAGE_KEYS)


# estimate_cost(): pre-run estimate used by --dry-run. Assumption: the shared system-prompt prefix
# is paid once at the cache-miss rate and at the cache-hit rate for the remaining n_calls - 1
# calls, while the per-ad user message is always a miss. This is why label_osca.py makes its first
# call alone before fanning out workers: if many workers start simultaneously before the cache is
# warm, more calls than one pay the miss rate and the estimate becomes optimistic. Retries and
# thinking-mode output are not included, so treat the number as a floor for those cases.
def estimate_cost(n_calls: int, prefix_tokens: int, per_call_input: int, per_call_output: int, model_key: str) -> float:
    """Rough estimate: the shared prefix is a cache miss once and a hit thereafter."""
    price = MODELS[model_key]["price"]
    return (prefix_tokens * price["cache_miss"] + (n_calls - 1) * prefix_tokens * price["cache_hit"]
            + n_calls * per_call_input * price["cache_miss"] + n_calls * per_call_output * price["output"]) / 1e6
