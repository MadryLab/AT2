from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from pathlib import Path
import torch as ch

from .attention import get_attention_weights, get_attentions_shape
from ..tasks import AttributionTask


class FeatureExtractor(ABC):
    def __init__(self, **kwargs: Dict[str, Any]):
        self.kwargs = kwargs

    _registry = {}

    def __init_subclass__(cls, **kwargs: Dict[str, Any]):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    @property
    @abstractmethod
    def num_features(self) -> int:
        """The number of features for the feature extractor."""

    @abstractmethod
    def __call__(
        self, task: AttributionTask, attribution_start: int, attribution_end: int
    ) -> ch.Tensor:
        """Extract features from the task."""

    def serialize(self):
        """Serialize the feature extractor."""
        data = {
            "class": self.__class__.__name__,
            "kwargs": self.kwargs,
        }
        return data

    @classmethod
    def deserialize(cls, data: Dict[str, Any]):
        """Deserialize the feature extractor."""
        class_name = data["class"]
        kwargs = data["kwargs"]
        feature_extractor_class = cls._registry.get(class_name)
        if feature_extractor_class is None:
            raise ValueError(f"Unknown feature extractor class: {class_name}")
        feature_extractor = feature_extractor_class(**kwargs)
        return feature_extractor


class AttentionFeatureExtractor(FeatureExtractor):
    def __init__(
        self, num_layers: int, num_heads: int, model_type: Optional[str] = None
    ):
        super().__init__(
            num_layers=num_layers, num_heads=num_heads, model_type=model_type
        )
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_type = model_type

    @classmethod
    def from_model(cls, model: Any) -> "AttentionFeatureExtractor":
        num_layers, num_heads = get_attentions_shape(model)
        return cls(num_layers, num_heads)

    @property
    def num_features(self) -> int:
        return self.num_layers * self.num_heads

    def __call__(
        self, task: AttributionTask, attribution_start: int, attribution_end: int
    ) -> ch.Tensor:
        # (num_layers, num_heads, num_target_tokens, num_tokens)
        weights = get_attention_weights(
            task.model,
            task.hidden_states,
            attribution_start=attribution_start,
            attribution_end=attribution_end,
            model_type=self.model_type,
        )
        # (num_target_tokens, num_tokens, num_layers, num_heads)
        weights = weights.permute(2, 3, 0, 1)
        num_target_tokens, num_tokens, _, _ = weights.shape
        # (num_target_tokens, num_tokens, num_layers * num_heads)
        weights = weights.view(num_target_tokens, num_tokens, -1)
        return weights
