"""
Evaluation engine computing ROC-AUC, Precision, Recall, F1, False Alarm Rate,
and damage localization accuracy.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from sklearn import metrics
import torch

from ..models.conv1d_bilstm_autoencoder import Conv1DBiLSTMAutoencoder
from ..models.health_index import StructuralHealthPredictor


@dataclass
class EvaluationResult:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1_score: float
    false_alarm_rate: float
    accuracy: float
    confusion_matrix: Dict[str, int]
    threshold: float
    reconstruction_errors: np.ndarray
    true_labels: np.ndarray
    predicted_labels: np.ndarray
    predicted_health_indices: np.ndarray


class ModelEvaluator:
    """
    Evaluates anomaly detection performance and structural health predictions.
    """

    def __init__(
        self,
        model: Conv1DBiLSTMAutoencoder,
        predictor: StructuralHealthPredictor,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.predictor = predictor
        self.device = device or next(model.parameters()).device

    def evaluate(
        self,
        test_windows: np.ndarray,
        true_labels: np.ndarray,
        true_shi: Optional[np.ndarray] = None,
        sensor_names: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate model on test dataset.
        test_windows: Shape [N, num_sensors, window_size] (already scaled)
        true_labels: Shape [N] (0 = healthy, 1 = anomalous)
        """
        self.model.eval()
        names = sensor_names or [f"S{i+1}" for i in range(test_windows.shape[1])]

        # Batch inference
        batch_size = 64
        num_windows = len(test_windows)
        all_win_errors = []
        all_sens_errors = []

        with torch.no_grad():
            for i in range(0, num_windows, batch_size):
                batch_tensor = torch.tensor(
                    test_windows[i : i + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                outputs = self.model(batch_tensor)
                all_win_errors.append(outputs["reconstruction_error"].cpu().numpy())
                all_sens_errors.append(outputs["sensor_errors"].cpu().numpy())

        win_errors = np.concatenate(all_win_errors, axis=0)
        sens_errors = np.concatenate(all_sens_errors, axis=0)

        # Health assessment
        assessments = self.predictor.assess_batch(win_errors, sens_errors, names)
        pred_labels = np.array([1 if a.is_anomaly else 0 for a in assessments], dtype=np.int64)
        pred_shi = np.array([a.health_index for a in assessments], dtype=np.float32)

        # Compute classification metrics
        y_true = np.array(true_labels, dtype=np.int64)

        # Handle edge cases (e.g. all 0 or all 1 in evaluation batch)
        if len(np.unique(y_true)) > 1:
            try:
                roc_auc = float(metrics.roc_auc_score(y_true, win_errors))
            except Exception:
                roc_auc = 0.5
            try:
                pr_auc = float(metrics.average_precision_score(y_true, win_errors))
            except Exception:
                pr_auc = float(np.mean(y_true))
        else:
            roc_auc = 1.0 if np.all(pred_labels == y_true) else 0.5
            pr_auc = 1.0 if np.all(pred_labels == y_true) else 0.5

        precision = float(metrics.precision_score(y_true, pred_labels, zero_division=0))
        recall = float(metrics.recall_score(y_true, pred_labels, zero_division=0))
        f1 = float(metrics.f1_score(y_true, pred_labels, zero_division=0))
        acc = float(metrics.accuracy_score(y_true, pred_labels))

        # Confusion Matrix
        try:
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, pred_labels).ravel()
            far = float(fp / max(1, (fp + tn)))
        except ValueError:
            # Fallback if only 1 class present
            tp = int(np.sum((pred_labels == 1) & (y_true == 1)))
            fp = int(np.sum((pred_labels == 1) & (y_true == 0)))
            fn = int(np.sum((pred_labels == 0) & (y_true == 1)))
            tn = int(np.sum((pred_labels == 0) & (y_true == 0)))
            far = float(fp / max(1, (fp + tn)))

        cm_dict = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

        return EvaluationResult(
            roc_auc=round(roc_auc, 4),
            pr_auc=round(pr_auc, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            false_alarm_rate=round(far, 4),
            accuracy=round(acc, 4),
            confusion_matrix=cm_dict,
            threshold=float(self.predictor.threshold),
            reconstruction_errors=win_errors,
            true_labels=y_true,
            predicted_labels=pred_labels,
            predicted_health_indices=pred_shi,
        )
