from pathlib import Path
import torch as ch
from fire import Fire
from datasets import load_dataset

from at2.utils import get_model_and_tokenizer
from at2.tasks import SimpleContextAttributionTask
from at2 import AT2Trainer
from scripts.utils import is_multimodal


def filter_fn(example):
    valid_category = example["category"] in [
        "summarization",
        "closed_qa",
        "information_extraction",
    ]
    valid_length = len(example["context"]) < 20_000
    return valid_category and valid_length


def task_from_example(example, model, tokenizer, source_type="token"):
    return SimpleContextAttributionTask(
        context=example["context"],
        query=example["instruction"],
        model=model,
        tokenizer=tokenizer,
        source_type=source_type,
    )


def main(
    command: str,
    model_name: str,
    job_index: int = 0,
    num_jobs: int = 1,
    num_examples: int = 4_000,
):
    raw_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    dataset = raw_dataset.filter(filter_fn).shuffle(seed=42).select(range(num_examples))

    dtype = ch.bfloat16
    attn_implementation = "flash_attention_2"
    model, tokenizer = get_model_and_tokenizer(
        model_name,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        is_multimodal=is_multimodal(model_name),
    )

    save_path = Path("outputs") / f"context_{model_name.split('/')[-1].lower()}"

    trainer = AT2Trainer(
        save_path=save_path,
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        task_from_example=task_from_example,
    )

    if command == "generate":
        trainer.generate(job_index=job_index, num_jobs=num_jobs, batch_size=2)
    elif command == "compute":
        trainer.compute_features_and_outputs(
            job_index=job_index, num_jobs=num_jobs, batch_size=2
        )
    elif command == "train":
        trainer.train(save_name="default")
    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    Fire(main)
