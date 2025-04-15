import string
from typing import List, Tuple, Optional, Any, Union
from pathlib import Path
import nltk
import numpy as np
import torch as ch
from spacy.lang.en import English
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForImageTextToText,
)


nltk.download("punkt_tab")


def get_model_and_tokenizer(
    model_name: str,
    torch_dtype: ch.dtype = ch.bfloat16,
    device: Optional[Union[str, ch.device]] = "cuda:0",
    attn_implementation: Optional[str] = None,
    is_multimodal: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    if is_multimodal:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            **kwargs,
        )
        model.language_model.name_or_path = model.name_or_path
        model.language_model.generation_config = model.generation_config
        model = model.language_model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            **kwargs,
        )
    if device is not None:
        model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def is_sentence(s):
    char_set = string.whitespace + string.digits + string.punctuation
    is_sentence_ = False
    for char in s:
        is_sentence_ |= not (char in char_set)
    return is_sentence_


def split_text(text: str, split_by: str) -> Tuple[List[str], List[str], List[str]]:
    """Split text into parts and return the parts, start indices, and separators."""
    parts = []
    separators = []
    indices = []

    if split_by == "sentence":
        for line in text.splitlines():
            for sentence in nltk.sent_tokenize(line):
                # Exclude things like "1." (these will be part of separator)
                if is_sentence(sentence):
                    parts.append(sentence)
    elif split_by == "word":
        tokenizer = English().tokenizer
        parts = [token.text for token in tokenizer(text) if len(token.text.strip()) > 0]
    else:
        raise ValueError(f"Cannot split text by '{split_by}'")

    cur_start = 0
    for part in parts:
        cur_end = text.find(part, cur_start)
        separator = text[cur_start:cur_end]
        separators.append(separator)
        cur_start = cur_end + len(part)
        indices.append((cur_end, cur_start))

    return parts, separators, indices


def get_job_start_and_end(
    num_examples: int,
    job_idx: int,
    num_jobs: Optional[int] = None,
    examples_per_job: Optional[int] = None,
):
    if examples_per_job is None:
        assert num_jobs is not None
        assert num_examples is not None
        examples_per_job = (num_examples + num_jobs - 1) // num_jobs
    else:
        assert num_jobs is None
        num_jobs = (num_examples + examples_per_job - 1) // examples_per_job
    start = job_idx * examples_per_job
    end = min((job_idx + 1) * examples_per_job, num_examples)
    return start, end


def infer_num_jobs(path: Path):
    files = list(path.iterdir())
    for file in files:
        if "0_of_" in file.name:
            return int(file.name.split("_of_")[1].split(".")[0])
    return None


def create_registry(size: int, num_jobs: Optional[int] = None):
    jobs = np.zeros((size,), dtype=np.int32)
    job_starts = np.zeros((num_jobs,), dtype=np.int32)
    if num_jobs is not None:
        for job_index in range(num_jobs):
            start, end = get_job_start_and_end(size, job_index, num_jobs)
            jobs[start:end] = job_index
            job_starts[job_index] = start
    return jobs, job_starts
