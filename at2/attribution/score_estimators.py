"""
Score estimators are learnable models that predict the attribution score for a given source.
In the case of AT2, the features are the attention weights of the generated sequence to the source tokens.
"""

import torch as ch
import torch.nn as nn
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from huggingface_hub import hf_hub_download

from .features import FeatureExtractor
from ..tasks import AttributionTask


class ScoreEstimator(nn.Module, ABC):
    """A learnable estimator of attribution scores."""

    def __init__(self, feature_extractor: FeatureExtractor, **kwargs: Dict[str, Any]):
        """Create a score estimator using the provided feature extractor."""
        super().__init__()
        self.feature_extractor = feature_extractor
        self.kwargs = kwargs

    _registry = {}

    def __init_subclass__(cls, **kwargs: Dict[str, Any]):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    @abstractmethod
    def project_parameters(self):
        """Project parameters into an allowed space (applied at each step)."""

    @abstractmethod
    def finalize_parameters(self):
        """Finalize parameters (applied after training)."""

    def get_scores(
        self, task: AttributionTask, attribution_start: int, attribution_end: int
    ):
        """Get scores by extracting features and passing them through the estimator."""
        features = self.feature_extractor(task, attribution_start, attribution_end)
        return self.forward(features)[:, :, 0]

    def save(self, path: Path, extras: Optional[Dict[str, Any]] = None):
        """Save the model to the specified path."""
        save_dict = {
            "class": self.__class__.__name__,
            "state_dict": self.state_dict(),
            "feature_extractor": self.feature_extractor.serialize(),
            "kwargs": self.kwargs,
            "extras": extras or {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        ch.save(save_dict, path)

    @classmethod
    def load(cls, path: Path, device: Optional[Union[str, ch.device]] = None):
        """Load a estimator from the specified path."""
        save_dict = ch.load(path, map_location=device, weights_only=False)
        class_name = save_dict["class"]
        state_dict = save_dict["state_dict"]
        kwargs = save_dict["kwargs"]
        extras = save_dict["extras"]
        feature_extractor = FeatureExtractor.deserialize(save_dict["feature_extractor"])
        estimator_class = cls._registry.get(class_name)
        if estimator_class is None:
            raise ValueError(f"Unknown estimator class: {class_name}")
        estimator = estimator_class(feature_extractor, **kwargs, extras=extras)
        estimator.load_state_dict(state_dict)
        return estimator

    @classmethod
    def from_hub(cls, model_name: str, device: Optional[Union[str, ch.device]] = None):
        """Load a score estimator from the specified model name on HuggingFace."""
        path = hf_hub_download(repo_id=model_name, filename="score_estimator.pt")
        return cls.load(Path(path), device)


class LinearScoreEstimator(ScoreEstimator):
    """A score estimator that predicts scores as a linear function of the features."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        normalize: bool = True,
        non_negative: bool = False,
        bias: bool = False,
        **kwargs: Dict[str, Any],
    ):
        """Create a linear score estimator using the provided feature extractor.

        Args:
            feature_extractor: The feature extractor to use.
            normalize: Whether to L1-normalize the weights (applied after training).
            non_negative: Whether to ensure the weights are non-negative (applied at every training step).
            bias: Whether to include a bias term.
        """
        super().__init__(
            feature_extractor,
            normalize=normalize,
            non_negative=non_negative,
            bias=bias,
            **kwargs,
        )
        num_features = feature_extractor.num_features
        self.linear = nn.Linear(num_features, 1, bias=bias)
        self.linear.weight.data[:] = 1 / num_features
        self.normalize = normalize
        self.non_negative = non_negative

    def forward(self, features):
        return self.linear(features)

    def project_parameters(self):
        if self.non_negative:
            self.linear.weight.data[:] = ch.clamp(self.linear.weight.data[:], min=0)

    def finalize_parameters(self):
        if self.normalize:
            self.linear.weight.data[:] /= ch.norm(self.linear.weight.data, p=1)
