import pytest
from pathlib import Path
import torch as ch
from datasets import load_dataset

from at2.tasks.context_attribution import SimpleContextAttributionTask
from at2.utils import get_model_and_tokenizer
from at2 import AT2Trainer, AT2ScoreEstimator


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
NUM_EXAMPLES = 64
BATCH_SIZE = 32
NUM_JOBS = 2


def is_multimodal(model_name):
    if "gemma-3-" in model_name.lower():
        params_start = model_name.lower().find("gemma-3-") + len("gemma-3-")
        params_end = model_name.lower().find("b-it", params_start)
        params = model_name[params_start:params_end]
        # Only gemma-3-1b-it is text-only, the rest are multimodal
        if params != "1":
            return True
    return False


def task_from_example(example, model, tokenizer, source_type="token"):
    context = example["article"]
    query = "Summarize the article in up to three sentences."

    task = SimpleContextAttributionTask(
        context=context,
        query=query,
        model=model,
        tokenizer=tokenizer,
        source_type=source_type,
    )
    return task


@pytest.fixture
def trainer(tmp_path: Path):
    save_path = tmp_path / "at2_trainer"
    raw_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
    dataset = raw_dataset.shuffle(seed=42).select(range(NUM_EXAMPLES))

    model, tokenizer = get_model_and_tokenizer(
        MODEL_NAME,
        torch_dtype=ch.bfloat16,
        is_multimodal=is_multimodal(MODEL_NAME),
    )

    trainer = AT2Trainer(
        save_path=save_path,
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        task_from_example=task_from_example,
    )
    return trainer


def test_generate(trainer: AT2Trainer):
    for job_index in range(NUM_JOBS):
        trainer.generate(job_index=job_index, num_jobs=NUM_JOBS, batch_size=2)


def test_compute(trainer: AT2Trainer):
    test_generate(trainer)
    for job_index in range(NUM_JOBS):
        trainer.compute_features_and_outputs(
            job_index=job_index, num_jobs=NUM_JOBS, batch_size=2
        )


def test_train(trainer: AT2Trainer):
    test_compute(trainer)
    trainer.train(save_name=f"default", batch_size=BATCH_SIZE)


def test_load_from_checkpoint(trainer: AT2Trainer):
    test_train(trainer)
    AT2ScoreEstimator.load(
        trainer.save_path / "estimators" / "default" / "score_estimator.pt"
    )
