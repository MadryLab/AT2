"""
Attributors perform attribution for a given task.
AT2 uses a score estimator to perform attribution.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch as ch


from ..tasks import AttributionTask
from .score_estimators import LinearScoreEstimator


DEFAULT_NUM_SOURCES_TO_SHOW = 8


class Attributor(ABC):
    def __init__(
        self,
        task: AttributionTask,
        cache: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any],
    ):
        self.task = task
        self._cache = cache or {}
        self._kwargs = kwargs

    @abstractmethod
    def prepare_for_attribution(self):
        """Compute any information needed for attribution across target ranges."""

    @abstractmethod
    def _get_attribution_scores_for_token_range(self, token_start, token_end, **kwargs):
        """Get attribution scores for a given token range."""

    def get_attribution_scores(
        self,
        start=None,
        end=None,
        token_start=None,
        token_end=None,
        verbose=False,
        **kwargs,
    ):
        """Get attribution scores for a given target range.
        The target range can be specified in terms of characters (`start` and `end`) or tokens (`token_start` and `token_end`).

        Args:
            start: The start index of the target range relative to the target text (in characters).
            end: The end index of the target range relative to the target text (in characters).
            token_start: The start token index of the target range relative to the target text (in tokens).
            token_end: The end token index of the target range relative to the target text (in tokens).
            verbose: Whether to display the attribution text.
        """

        if start is not None or end is not None:
            assert token_start is None and token_end is None
            if self.task.num_sources == 0:
                return np.array([])
            token_start, token_end = self.task.target_range_to_token_range(
                start_index=start, end_index=end, relative=True
            )
            selected = self.task.target[start:end]
            attributed = self.task.tokenizer.decode(
                self.task.target_ids[token_start:token_end]
            )
            if verbose:
                print("Computing attribution scores for:\n", attributed.strip())
            if selected.strip() not in attributed.strip():
                print(
                    f'Warning: selected text "{selected.strip()}" not in attributed text "{attributed.strip()}"'
                )
        token_start = token_start if token_start is not None else 0
        token_end = token_end if token_end is not None else len(self.task.target_ids)
        return self._get_attribution_scores_for_token_range(
            token_start, token_end, **kwargs
        )

    def show_attribution(
        self,
        start=None,
        end=None,
        token_start=None,
        token_end=None,
        num_sources_to_show=DEFAULT_NUM_SOURCES_TO_SHOW,
        cmap="RdBu",
        **kwargs,
    ):
        """Show the attribution scores in a styled dataframe."""

        scores = self.get_attribution_scores(
            start=start, end=end, token_start=token_start, token_end=token_end, **kwargs
        )
        applicable_indices = np.where(~np.isnan(scores))[0]
        order = applicable_indices[scores[applicable_indices].argsort()[::-1]]
        scores_and_sources = []
        for i in order[:num_sources_to_show]:
            scores_and_sources.append((scores[i], self.task.get_source(i)))
        df = pd.DataFrame(scores_and_sources, columns=["Score", "Source"])
        styled_df = df.style
        max_score = df["Score"].abs().max() + 1e-6
        styled_df = styled_df.background_gradient(
            cmap=cmap, subset=["Score"], vmin=-max_score, vmax=max_score
        )
        styled_df = styled_df.format(precision=3)
        return styled_df

    def highlight_attribution(
        self,
        start=None,
        end=None,
        token_start=None,
        token_end=None,
        max_score=None,
    ):
        """Highlight the attribution scores in the task text."""

        if token_start is None or token_end is None:
            token_start, token_end = self.task.target_range_to_token_range(
                start, end, relative=True
            )
        target_token_start, _ = self.task.target_token_range
        source_scores = self.get_attribution_scores(
            token_start=token_start, token_end=token_end
        )
        source_scores[np.isnan(source_scores)] = 0
        if max_score is not None:
            source_scores = np.clip(source_scores, -max_score, max_score)
        else:
            max_score = np.abs(source_scores).max() + 1e-6
        source_scores = source_scores / max_score

        tokens = self.task.get_tokens()
        token_scores = np.zeros((len(tokens["input_ids"]),))
        for i, (source_token_start, source_token_end) in enumerate(
            self.task.source_token_ranges
        ):
            token_scores[source_token_start:source_token_end] = source_scores[i]

        attribution_text = ""
        for i in range(len(tokens["input_ids"])):
            token_text_range = tokens.token_to_chars(i)
            token_text = self.task.text[token_text_range.start : token_text_range.end]

            if i >= (target_token_start + token_start) and i < (
                target_token_start + token_end
            ):
                highlighted_token = f"\033[38;2;255;255;0m{token_text}\033[0m"
            else:
                score = token_scores[i].item()
                color_intensity = int(255 * np.abs(score))
                if score > 0:
                    highlighted_token = f"\033[38;2;{255-color_intensity};{255-color_intensity};255m{token_text}\033[0m"
                else:
                    highlighted_token = f"\033[38;2;255;{255-color_intensity};{255-color_intensity}m{token_text}\033[0m"
            attribution_text += highlighted_token
        return attribution_text


class ScoreEstimationAttributor(Attributor):
    def __init__(
        self,
        task: AttributionTask,
        score_estimator: LinearScoreEstimator,
        cache: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            task,
            cache=cache,
        )
        self.score_estimator = score_estimator
        self.score_estimator = self.score_estimator.to(self.task.model.device)
        self.score_estimator = self.score_estimator.to(self.task.model.dtype)

    @classmethod
    def from_path(cls, task: AttributionTask, path: Path):
        score_estimator = LinearScoreEstimator.load(path)
        return cls(task, score_estimator)

    @classmethod
    def from_hub(cls, task: AttributionTask, model_name: str):
        score_estimator = LinearScoreEstimator.from_hub(model_name)
        return cls(task, score_estimator)

    def _get_scores(self):
        with ch.no_grad():
            scores = self.score_estimator.get_scores(
                self.task, *self.task.target_token_range
            )
        scores = [scores[:, s:e].sum(dim=-1) for s, e in self.task.source_token_ranges]
        scores = ch.stack(scores, dim=0)  # [num_sources x num_target_tokens]
        return scores

    @property
    def scores(self):
        if self._cache.get("scores") is None:
            self._cache["scores"] = self._get_scores()
        return self._cache["scores"]

    def prepare_for_attribution(self):
        self.scores

    def _get_attribution_scores_for_token_range(self, token_start, token_end, **kwargs):
        aggregated_scores = self.scores[:, token_start:token_end].mean(dim=-1)
        return aggregated_scores.cpu().type(ch.float32).numpy()
