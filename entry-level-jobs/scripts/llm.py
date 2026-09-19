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
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import TypeVar

import openai
from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_env  # noqa: E402

BASE_URL = "https://api.deepseek.com"

# USD per million tokens at PEAK rates (01:00-04:00 and 06:00-10:00 UTC, Mon-Fri). Off-peak is
# half of these. Cost summaries printed by the scripts use peak rates, so they are upper bounds.
MODELS = {
    "flash": {"id": "deepseek-flash",  "price": {"cache_hit": 0.006, "cache_miss": 0.30, "output": 1.20}},
    "pro":   {"id": "deepseek-v4-pro", "price": {"cache_hit": 0.044, "cache_miss": 1.32, "output": 3.96}},
}
DEFAULT_MODEL = "flash"
USAGE_KEYS = ("cache_hit", "cache_miss", "output")

T = TypeVar("T", bound=BaseModel)


class Refusal(Exception):
    """Raised when the model returns nothing usable after all retries."""


_client: openai.OpenAI | None = None


def client() -> openai.OpenAI:
    global _client
    if _client is None:
        load_env()
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            sys.exit("DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add your key.")
        _client = openai.OpenAI(api_key=key, base_url=BASE_URL, max_retries=4, timeout=600)
    return _client


def schema_instruction(schema: type[BaseModel]) -> str:
    """Text appended to the system prompt so JSON mode knows the shape to produce."""
    return (
        "\n\nRespond with a single JSON object and nothing else. It must validate against this JSON schema:\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        + "\nUse exactly the field names shown. Do not wrap the object in markdown."
    )


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
    messages = [
        {"role": "system", "content": system + schema_instruction(schema)},
        {"role": "user", "content": user},
    ]
    usage_total = {k: 0 for k in USAGE_KEYS}
    last_error = ""
    for attempt in range(retries + 1):
        resp = client().chat.completions.create(
            model=model["id"],
            messages=messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        )
        u = resp.usage
        usage_total["cache_hit"] += getattr(u, "prompt_cache_hit_tokens", 0) or 0
        usage_total["cache_miss"] += getattr(u, "prompt_cache_miss_tokens", None) or (u.prompt_tokens - (getattr(u, "prompt_cache_hit_tokens", 0) or 0))
        usage_total["output"] += u.completion_tokens or 0
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        if choice.finish_reason == "length":
            last_error = "response was cut off by max_tokens"
        elif not content:
            last_error = "empty response"  # documented DeepSeek JSON-mode failure mode
        else:
            try:
                return schema.model_validate(json.loads(content)), usage_total, resp.model
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = f"{type(e).__name__}: {str(e)[:500]}"
        if attempt < retries:
            # feed the failure back so the retry can correct it
            messages = messages[:2] + [
                {"role": "assistant", "content": content or "(empty)"},
                {"role": "user", "content": f"That response was invalid ({last_error}). Return only the JSON object, complete and valid."},
            ]
            time.sleep(1)
    raise Refusal(last_error)


def cost(usage: dict[str, int], model_key: str) -> float:
    price = MODELS[model_key]["price"]
    return sum(usage.get(k, 0) * price[k] / 1e6 for k in USAGE_KEYS)


def estimate_cost(n_calls: int, prefix_tokens: int, per_call_input: int, per_call_output: int, model_key: str) -> float:
    """Rough estimate: the shared prefix is a cache miss once and a hit thereafter."""
    price = MODELS[model_key]["price"]
    return (prefix_tokens * price["cache_miss"] + (n_calls - 1) * prefix_tokens * price["cache_hit"]
            + n_calls * per_call_input * price["cache_miss"] + n_calls * per_call_output * price["output"]) / 1e6
