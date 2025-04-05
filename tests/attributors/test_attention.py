import pytest
from typing import Any, Dict, Tuple
import torch as ch
from transformers import AutoModelForCausalLM, AutoTokenizer

from at2.tasks.context_attribution import SimpleContextAttributionTask
from at2.attribution.attention import get_attention_weights


MODEL_NAMES = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]


def get_model_and_tokenizer(model_name: str) -> tuple[Any, Any]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=ch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", trust_remote_code=True
    )
    return model, tokenizer


@pytest.fixture(scope="module")
def models_and_tokenizers() -> Dict[str, Tuple[Any, Any]]:
    return {
        model_name: get_model_and_tokenizer(model_name) for model_name in MODEL_NAMES
    }


def get_task(model: Any, tokenizer: Any) -> SimpleContextAttributionTask:
    task = SimpleContextAttributionTask(
        context="Cacti can develop root rot if you water them too much.",
        query="Can you overwater a cactus? Please respond in a single sentence.",
        model=model,
        tokenizer=tokenizer,
    )
    return task


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_get_attention_weights(
    model_name: str,
    models_and_tokenizers: Dict[str, Tuple[Any, Any]],
) -> None:
    """Test that the attention weights computed by our get_attention_weights
    match the attention weights computed by the transformers implementation.
    """
    model, tokenizer = models_and_tokenizers[model_name]
    task = get_task(model, tokenizer)
    tokens_pt = task.get_tokens(return_tensors="pt").to(model.device)
    with ch.no_grad():
        output = model(**tokens_pt, output_hidden_states=True, output_attentions=True)
    hidden_states = [output.hidden_states[i][:, :-1] for i in range(len(output.hidden_states))]
    weights = get_attention_weights(model, hidden_states)[-1]
    transformers_weights = output.attentions[-1][:, :, :-1, :-1]
    assert ch.allclose(weights, transformers_weights)
