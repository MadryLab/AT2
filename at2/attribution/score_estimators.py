import torch as ch
import torch.nn as nn
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union


class ScoreEstimator(nn.Module, ABC):
    """A learnable estimator of attribution scores."""

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__()
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

    def save(self, path: Path, extras: Optional[Dict[str, Any]] = None):
        """Save the model to the specified path."""
        save_dict = {
            "class": self.__class__.__name__,
            "state_dict": self.state_dict(),
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
        estimator_class = cls._registry.get(class_name)
        if estimator_class is None:
            raise ValueError(f"Unknown estimator class: {class_name}")
        estimator = estimator_class(**kwargs, extras=extras)
        estimator.load_state_dict(state_dict)
        return estimator


class LinearScoreEstimator(ScoreEstimator):
    def __init__(
        self,
        num_features: int,
        normalize: bool = True,
        non_negative: bool = False,
        bias: bool = False,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(
            num_features=num_features,
            normalize=normalize,
            non_negative=non_negative,
            bias=bias,
            **kwargs,
        )
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
