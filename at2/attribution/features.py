from abc import ABC, abstractmethod
from typing import Any
import torch as ch

from .attention import get_attention_weights, get_attentions_shape
from ..tasks import AttributionTask


class FeatureExtractor(ABC):
    @property
    @abstractmethod
    def num_features(self) -> int:
        """The number of features for the feature extractor."""

    @abstractmethod
    def __call__(
        self, task: AttributionTask, attribution_start: int, attribution_end: int
    ) -> ch.Tensor:
        """Extract features from the task."""


class AttentionFeatureExtractor(FeatureExtractor):
    def __init__(self, model: Any):
        self.num_layers, self.num_heads = get_attentions_shape(model)

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
        )
        # (num_target_tokens, num_tokens, num_layers, num_heads)
        weights = weights.permute(2, 3, 0, 1)
        num_target_tokens, num_tokens, _, _ = weights.shape
        # (num_target_tokens, num_tokens, num_layers * num_heads)
        weights = weights.view(num_target_tokens, num_tokens, -1)
        return weights
