"""
Training loops, optimization routines, and evaluation metrics for bridge SHM.
"""
from .trainer import ModelTrainer, TrainingResult
from .evaluator import ModelEvaluator, EvaluationResult

__all__ = ["ModelTrainer", "TrainingResult", "ModelEvaluator", "EvaluationResult"]
