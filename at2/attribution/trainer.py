"""
The trainer is responsible for training a score estimator.
This implementation can only train linear score estimators.
"""

import json
import numpy as np
import torch as ch
from pathlib import Path
from tqdm.auto import tqdm
from numpy.lib.format import open_memmap
from datasets import Dataset
from typing import Optional, Dict, Any, Tuple, List, Callable


from ..tasks import AttributionTask
from ..utils import (
    get_job_start_and_end,
    infer_num_jobs,
    create_registry,
)
from .utils import aggregate_logit_probs
from .score_estimators import LinearScoreEstimator
from .features import FeatureExtractor, AttentionFeatureExtractor


class PearsonCorrelationLoss(ch.nn.Module):
    """A PyTorch loss function for the negative Pearson correlation."""

    def __init__(self, reduction: str = "mean"):
        super(PearsonCorrelationLoss, self).__init__()
        self.reduction = reduction

    def forward(self, predictions, targets):
        pred_mean = predictions.mean(dim=1, keepdim=True)  # [batch_size, 1]
        target_mean = targets.mean(dim=1, keepdim=True)  # [batch_size, 1]

        pred_diff = predictions - pred_mean  # [batch_size, sample_size]
        target_diff = targets - target_mean  # [batch_size, sample_size]

        covariance = (pred_diff * target_diff).mean(dim=1)  # [batch_size]
        pred_std = ch.sqrt((pred_diff**2).mean(dim=1))  # [batch_size]
        target_std = ch.sqrt((target_diff**2).mean(dim=1))  # [batch_size]

        correlation = covariance / (pred_std * target_std + 1e-8)  # [batch_size]
        losses = -correlation
        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "none":
            return losses
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class LinearScoreEstimatorTrainer:
    def __init__(
        self,
        save_path: Path,
        dataset: Dataset,
        model: Any,
        tokenizer: Any,
        task_from_example: Callable[[Dict[str, Any], Any, Any], AttributionTask],
        feature_extractor: Optional[FeatureExtractor] = None,
        split_target_by: Optional[str] = "sentence",
        num_masks: int = 32,
        generations_save_path: Optional[Path] = None,
    ):
        """Create a score estimator trainer.

        Args:
            save_path: The path to save the trainer to.
            dataset: The dataset to train on.
            model: The model to train a score estimator for.
            tokenizer: The tokenizer to train a score estimator for.
            task_from_example: A function that creates an attribution task from an example of the dataset.
            feature_extractor: The feature extractor to use.
            split_target_by: The unit of the generation to attribute.
            num_masks: The number of masks to use for each example.
            generations_save_path: The path to save the generations to.
        """
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.generations_save_path = (
            self.save_path / "generations"
            if generations_save_path is None
            else generations_save_path
        )
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = (
            AttentionFeatureExtractor.from_model(model)
            if feature_extractor is None
            else feature_extractor
        )
        self.task_from_example = task_from_example
        self.num_masks = num_masks
        self.split_target_by = split_target_by
        self._generations = None
        self._task_datas = None
        self._registry = None
        self._stores = None

    def get_task(self, i: int, load_generation: bool = True):
        example = self.dataset[i]
        task = self.task_from_example(example, self.model, self.tokenizer)
        if load_generation:
            task.set_generation(self.generations[i])
        return task

    def load_multiple_jsons(self, path: Path, require_complete=True):
        data = {}
        num_files = len(list(path.iterdir()))
        for i in range(num_files):
            file_path = path / f"{i}_of_{num_files}.json"
            with open(file_path, "r") as f:
                cur_data = json.load(f)
            cur_data = {int(k): v for k, v in cur_data.items()}
            data.update(cur_data)
        if len(data) != len(self.dataset) and require_complete:
            return None
        return data

    def load_generations(self, require_complete=True):
        return self.load_multiple_jsons(
            self.generations_save_path, require_complete=require_complete
        )

    @property
    def task_datas_save_path(self):
        return self.save_path / "task_datas"

    def load_task_datas(self, require_complete=True):
        return self.load_multiple_jsons(
            self.task_datas_save_path, require_complete=require_complete
        )

    @property
    def generations(self):
        if self._generations is None:
            self._generations = self.load_generations()
        return self._generations

    @property
    def task_datas(self):
        if self._task_datas is None:
            self._task_datas = self.load_task_datas()
        return self._task_datas

    def generate(self, job_index: int = 0, num_jobs: int = 1, batch_size: int = 4):
        """Generate completions for a subset of the dataset.

        Args:
            job_index: The index of the job to generate completions for.
            num_jobs: The total number of jobs to generate completions for.
            batch_size: The batch size to use.
        """

        def create_dict_from_json(base_path: Path):
            base_path.mkdir(parents=True, exist_ok=True)
            path = base_path / f"{job_index}_of_{num_jobs}.json"
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                    data = {int(k): v for k, v in data.items()}
            else:
                data = {}
            return data, path

        generations, cur_generations_save_path = create_dict_from_json(
            self.generations_save_path
        )
        task_datas, cur_task_datas_save_path = create_dict_from_json(
            self.task_datas_save_path
        )
        start, end = get_job_start_and_end(len(self.dataset), job_index, num_jobs)
        for cur_start in tqdm(
            range(start, end, batch_size), desc="Generating completions"
        ):
            cur_end = min(cur_start + batch_size, end)
            cur_indices = range(cur_start, cur_end)
            generations_incomplete = any(
                example_index not in generations for example_index in cur_indices
            )
            task_datas_incomplete = any(
                example_index not in task_datas for example_index in cur_indices
            )
            if generations_incomplete or task_datas_incomplete:
                tasks = [
                    self.get_task(example_index, load_generation=False)
                    for example_index in cur_indices
                ]
                if generations_incomplete:
                    cur_generations = AttributionTask.batch_generate(tasks)
                    for i, generation in zip(cur_indices, cur_generations):
                        generations[i] = generation
                    with open(cur_generations_save_path, "w") as f:
                        json.dump(generations, f, indent=4)
                else:
                    for i, task in zip(cur_indices, tasks):
                        task.set_generation(generations[i])
                if task_datas_incomplete:
                    for i, task in zip(cur_indices, tasks):
                        target_token_ranges = task.get_sub_target_token_ranges(
                            split_by=self.split_target_by, relative=True
                        )
                        task_datas[i] = {
                            "num_sources": task.num_sources,
                            "num_targets": len(target_token_ranges),
                        }
                    with open(cur_task_datas_save_path, "w") as f:
                        json.dump(task_datas, f, indent=4)

    @property
    def num_features(self):
        return self.feature_extractor.num_features

    @property
    def features_and_outputs_save_path(self):
        return self.save_path / "features_and_outputs"

    def get_store_for_job(
        self,
        job_index: Optional[int] = None,
        num_jobs: Optional[int] = None,
    ):
        if job_index is not None and num_jobs is not None:
            start, end = get_job_start_and_end(len(self.dataset), job_index, num_jobs)
            suffix = f"_{job_index}_of_{num_jobs}"
        else:
            start, end = 0, len(self.dataset)
            suffix = ""
        self.features_and_outputs_save_path.mkdir(parents=True, exist_ok=True)
        features_path = self.features_and_outputs_save_path / f"features{suffix}.npy"
        outputs_path = self.features_and_outputs_save_path / f"outputs{suffix}.npy"
        completed_path = self.features_and_outputs_save_path / f"completed{suffix}.npy"
        num_total_targets = 0
        num_examples = end - start
        example_starts = np.zeros((num_examples,), dtype=np.int32)
        for i in range(start, end):
            task_data = self.task_datas[i]
            num_targets = task_data["num_targets"]
            example_starts[i - start] = num_total_targets
            num_total_targets += num_targets
        if outputs_path.exists():
            features = np.load(features_path, mmap_mode="r+")
            outputs = np.load(outputs_path, mmap_mode="r+")
            completed = np.load(completed_path, mmap_mode="r+")
        else:
            features = open_memmap(
                features_path,
                dtype=np.float16,
                mode="w+",
                shape=(
                    num_total_targets,
                    self.num_masks,
                    self.num_features,
                ),
            )
            outputs = open_memmap(
                outputs_path,
                dtype=np.float16,
                mode="w+",
                shape=(num_total_targets, self.num_masks),
            )
            completed = open_memmap(
                completed_path, dtype=np.bool_, mode="w+", shape=(num_examples,)
            )
            completed[:] = False
            completed.flush()
        return features, outputs, completed, example_starts

    def compute_features_and_outputs_for_example(
        self,
        i: int,
        batch_size: int = 4,
    ):
        task = self.get_task(i)
        target_token_ranges = task.get_sub_target_token_ranges(
            split_by=self.split_target_by,
            relative=True,
        )
        num_targets = len(target_token_ranges)
        assert self.task_datas[i]["num_sources"] == task.num_sources
        assert self.task_datas[i]["num_targets"] == num_targets

        # Compute raw features for every source and target
        device = self.model.device
        target_token_start, target_token_end = task.target_token_range
        masks, logit_probs = task.get_masks_and_logit_probs(
            num_masks=self.num_masks,
            alpha=0.5,
            batch_size=batch_size,
            base_seed=0,
        )

        # (num_target_tokens, num_tokens, num_features)
        features = self.feature_extractor(task, target_token_start, target_token_end)
        _, num_tokens, num_features = features.shape
        if self.split_target_by == "token":
            features_by_target = features
        else:
            features_by_target = features.new_zeros(
                (num_targets, num_tokens, num_features)
            )
            for i, (s, e) in enumerate(target_token_ranges):
                features_by_target[i, :, :] = features[s:e, :, :].mean(dim=0)
            del features
        features_by_target_source = features_by_target.new_zeros(
            (num_targets, task.num_sources, num_features),
        )
        for i, (s, e) in enumerate(task.source_token_ranges):
            features_by_target_source[:, i, :] = features_by_target[:, s:e, :].sum(
                dim=1
            )
        del features_by_target

        features_by_target_mask = features_by_target_source.new_zeros(
            (num_targets, self.num_masks, self.num_features),
        )
        outputs = np.zeros((num_targets, self.num_masks))
        dtype = features_by_target_source.dtype
        masks = ch.tensor(masks, device=device, dtype=dtype) * 2 - 1
        for i, (token_start, token_end) in enumerate(target_token_ranges):
            features_by_target_mask[i] = masks @ features_by_target_source[i]
            outputs[i] = aggregate_logit_probs(logit_probs[:, token_start:token_end])
        features = features_by_target_mask.cpu().type(ch.float16).numpy()
        return features, outputs

    def compute_features_and_outputs(
        self,
        job_index: int = 0,
        num_jobs: int = 1,
        batch_size: int = 4,
    ):
        """Compute features and outputs for a subset of the dataset.

        Args:
            job_index: The index of the job to compute features and outputs for.
            num_jobs: The total number of jobs to compute features and outputs for.
            batch_size: The batch size to use.
        """
        features, outputs, completed, example_starts = self.get_store_for_job(
            job_index, num_jobs
        )
        start, end = get_job_start_and_end(len(self.task_datas), job_index, num_jobs)
        for i in tqdm(range(start, end), desc="Computing features and outputs"):
            store_i = i - start
            example_start = example_starts[store_i]
            task_data = self.task_datas[i]
            num_targets = task_data["num_targets"]
            example_end = example_start + num_targets
            if (
                completed[store_i]
                and np.any(features[example_start:example_end] != 0)
                and np.any(outputs[example_start:example_end] != 0)
            ):
                continue
            cur_features, cur_outputs = self.compute_features_and_outputs_for_example(
                i, batch_size=batch_size
            )
            features[example_start:example_end] = cur_features
            outputs[example_start:example_end] = cur_outputs
            features.flush()
            outputs.flush()
            completed[store_i] = True
            completed.flush()

    def get_features_and_outputs(self, i: int):
        if self._registry is None:
            num_jobs = infer_num_jobs(self.features_and_outputs_save_path)
            self._registry = create_registry(len(self.dataset), num_jobs)
            if num_jobs is None:
                self._stores = [self.get_store_for_job()]
            else:
                self._stores = []
                for job_index in range(num_jobs):
                    self._stores.append(
                        self.get_store_for_job(job_index=job_index, num_jobs=num_jobs)
                    )
        jobs, job_starts = self._registry
        job_index = jobs[i]
        features, outputs, completed, example_starts = self._stores[job_index]
        store_i = i - job_starts[job_index]
        if not completed[store_i]:
            return None, None
        task_data = self.task_datas[i]
        num_targets = task_data["num_targets"]
        example_start = example_starts[store_i]
        example_end = example_start + num_targets
        if store_i < len(completed) - 1:
            next_example_start = example_starts[store_i + 1]
            assert example_end == next_example_start
        cur_features = features[example_start:example_end]
        cur_outputs = outputs[example_start:example_end]
        return cur_features, cur_outputs

    def is_features_and_outputs_completed(self, i: int):
        features, _ = self.get_features_and_outputs(i)
        return features is not None

    def _sample_batch(
        self,
        batch_size: int,
        example_indices: List[int],
        instances: List[Tuple[int, int]],
        device: str,
        sampling_method: str = "examples",
        dtype: ch.dtype = ch.float32,
    ):
        # sampling_method specifies whether to sample uniformly across examples or instances
        if sampling_method == "examples":
            sampled_example_indices = np.random.choice(
                example_indices, batch_size, replace=False
            )
            sampled_target_indices = [
                np.random.randint(self.task_datas[i]["num_targets"])
                for i in sampled_example_indices
            ]
            sampled_instances = list(
                zip(sampled_example_indices, sampled_target_indices)
            )
        elif sampling_method == "instances":
            sampled_indices = np.random.choice(
                len(instances), batch_size, replace=False
            )
            sampled_instances = [instances[i] for i in sampled_indices]
        batch_features = np.zeros(
            (batch_size, self.num_masks, self.num_features),
            dtype=np.float16,
        )
        batch_outputs = np.zeros((batch_size, self.num_masks), dtype=np.float16)
        for batch_index, (i, target_index) in enumerate(sampled_instances):
            cur_features, cur_outputs = self.get_features_and_outputs(i)
            batch_features[batch_index] = cur_features[target_index]
            batch_outputs[batch_index] = cur_outputs[target_index]
        batch_features = ch.tensor(batch_features, device=device, dtype=dtype)
        batch_outputs = ch.tensor(batch_outputs, device=device, dtype=dtype)
        return batch_features, batch_outputs

    def train(
        self,
        score_estimator: Optional[LinearScoreEstimator] = None,
        num_iterations: int = 1_000,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        loss_fn: Optional[ch.nn.Module] = None,
        max_grad_norm: float = 1.0,
        sampling_method: str = "examples",
        train_indices: Optional[List[int]] = None,
        dtype: ch.dtype = ch.float32,
        device: str = "cuda:0",
        print_every: int = 100,
        save_name: Optional[str] = None,
    ):
        """Train a score estimator.

        Args:
            score_estimator: The score estimator to train.
            num_iterations: The number of iterations to train for.
            learning_rate: The learning rate to use.
            batch_size: The batch size to use.
            loss_fn: The loss function to use (by default, negative Pearson correlation).
            max_grad_norm: The maximum gradient norm to use for clipping.
            sampling_method: The sampling method to use (examples or instances).
            train_indices: The indices of the examples to train on (by default, all examples with completed features and outputs).
            dtype: The data type to use.
            device: The device to use.
            print_every: The number of iterations to print the loss.
            save_name: The name of the directory to save the score estimator and feature extractor to (relative to the save path).
        """
        if loss_fn is None:
            loss_fn = PearsonCorrelationLoss()
        if train_indices is None:
            train_indices = [
                i
                for i in range(len(self.dataset))
                if self.is_features_and_outputs_completed(i)
            ]
        # Exclude examples without at least one target
        train_indices = [
            i for i in train_indices if self.task_datas[i]["num_targets"] > 0
        ]
        print(f"Training on {len(train_indices)} examples of {len(self.dataset)}")
        train_instances = []
        for i in train_indices:
            for target_index in range(self.task_datas[i]["num_targets"]):
                train_instances.append((i, target_index))
        if score_estimator is None:
            score_estimator = LinearScoreEstimator(self.feature_extractor).to(device)
        else:
            score_estimator = score_estimator.to(device)
        optimizer = ch.optim.Adam(score_estimator.parameters(), lr=learning_rate)
        scheduler = ch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iterations
        )
        cur_losses = []
        for step in tqdm(range(num_iterations), desc="Training score estimator"):
            batch_features, batch_outputs = self._sample_batch(
                batch_size,
                train_indices,
                train_instances,
                device,
                sampling_method=sampling_method,
                dtype=dtype,
            )
            optimizer.zero_grad()
            predictions = score_estimator(batch_features)[:, :, 0]
            loss = loss_fn(predictions, batch_outputs)
            loss.backward()
            ch.nn.utils.clip_grad_norm_(
                score_estimator.parameters(), max_norm=max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            cur_losses.append(loss.item())
            if step == 0 or (step + 1) % print_every == 0:
                mean_loss = np.mean(cur_losses)
                print(f"Step {step}: loss={mean_loss:.4g}")
                cur_losses = []
            score_estimator.project_parameters()
        score_estimator.finalize_parameters()
        if save_name is not None:
            save_path = self.save_path / "estimators" / f"{save_name}"
            save_path.mkdir(parents=True, exist_ok=True)
            score_estimator.save(save_path / "score_estimator.pt")
            print(f"Saved estimator to {save_path}")
        return score_estimator
