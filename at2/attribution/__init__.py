from .trainer import LinearScoreEstimatorTrainer
from .features import FeatureExtractor, AttentionFeatureExtractor
from .score_estimators import ScoreEstimator, LinearScoreEstimator
from .attributors import ScoreEstimationAttributor

AT2Trainer = LinearScoreEstimatorTrainer
AT2Attributor = ScoreEstimationAttributor
AT2FeatureExtractor = AttentionFeatureExtractor
AT2ScoreEstimator = LinearScoreEstimator
