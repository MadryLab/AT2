import pytest
from typing import Any
import numpy as np
import torch as ch

from at2.tasks.context_attribution import SimpleContextAttributionTask
from at2.tasks.thought_attribution import SimpleThoughtAttributionTask
from at2.utils import get_model_and_tokenizer
from at2 import AT2Attributor


CONTEXT_MODEL_NAMES = [
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

THOUGHT_MODEL_NAMES = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]


def get_context_task(model: Any, tokenizer: Any) -> SimpleContextAttributionTask:
    task = SimpleContextAttributionTask(
        context="Cacti can develop root rot if you water them too much.",
        query="Can you overwater a cactus? Please respond in a single sentence.",
        model=model,
        tokenizer=tokenizer,
        source_type="token",
    )
    return task


def get_thought_task(model: Any, tokenizer: Any) -> SimpleThoughtAttributionTask:
    task = SimpleThoughtAttributionTask(
        query="Can you overwater a cactus? Please respond in a single sentence.",
        model=model,
        tokenizer=tokenizer,
        source_type="token",
    )
    return task


def is_multimodal(model_name):
    if "gemma-3-" in model_name.lower():
        params_start = model_name.lower().find("gemma-3-") + len("gemma-3-")
        params_end = model_name.lower().find("b-it", params_start)
        params = model_name[params_start: params_end]
        # Only gemma-3-1b-it is text-only, the rest are multimodal
        if params != "1":
            return True
    return False


def get_hub_name(model_name: str) -> str:
    return f"madrylab/at2-{model_name.split('/')[-1].lower()}"


@pytest.mark.parametrize("model_name", CONTEXT_MODEL_NAMES)
def test_context_attributor_from_hub(model_name: str) -> None:
    model, tokenizer = get_model_and_tokenizer(
        model_name, dtype=ch.bfloat16, is_multimodal=is_multimodal(model_name)
    )
    task = get_context_task(model, tokenizer)
    hub_name = get_hub_name(model_name)
    attributor = AT2Attributor.from_hub(task, hub_name)
    scores = attributor.get_attribution_scores()
    assert isinstance(scores, np.ndarray)


@pytest.mark.parametrize("model_name", THOUGHT_MODEL_NAMES)
def test_thought_attributor_from_hub(model_name: str) -> None:
    model, tokenizer = get_model_and_tokenizer(
        model_name, dtype=ch.bfloat16, is_multimodal=is_multimodal(model_name)
    )
    task = get_thought_task(model, tokenizer)
    hub_name = get_hub_name(model_name)
    attributor = AT2Attributor.from_hub(task, hub_name)
    scores = attributor.get_attribution_scores()
    assert isinstance(scores, np.ndarray)
