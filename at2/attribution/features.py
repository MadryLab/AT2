"""
Feature extractors produce features relevant for attribution (estimating the influence of a source on a model's generation).
In the case of AT2, the features are the attention weights of the generated sequence to the source tokens.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch as ch

from .attention import get_attention_weights, get_attentions_shape
from ..tasks import AttributionTask


class FeatureExtractor(ABC):
    """A feature extractor that produces features relevant for attribution."""

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
        """Create an attention feature extractor.

        Args:
            num_layers: The number of layers in the model.
            num_heads: The number of heads in the model.
            model_type: The type of model (used by `get_attention_weights`).
        """
        super().__init__(
            num_layers=num_layers, num_heads=num_heads, model_type=model_type
        )
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_type = model_type

    @classmethod
    def from_model(cls, model: Any) -> "AttentionFeatureExtractor":
        """Create an attention feature extractor from a model."""
        num_layers, num_heads = get_attentions_shape(model)
        return cls(num_layers, num_heads)

    @property
    def num_features(self) -> int:
        return self.num_layers * self.num_heads

    def __call__(
        self, task: AttributionTask, attribution_start: int, attribution_end: int
    ) -> ch.Tensor:
        """Extract attention weights from the task as features for attribution."""
        # This computes attention weights from hidden states only for the specified
        # attribution range (this is more efficient than computing the entire attention
        # matrix using the transformers implementation).
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
        # (num_target_tokens, num_tokens, num_features)
        weights = weights.view(num_target_tokens, num_tokens, self.num_features)
        return weights
