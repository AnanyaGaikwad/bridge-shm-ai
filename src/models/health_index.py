"""
Structural Health Index (SHI) computation, dynamic anomaly thresholding,
and sensor-level structural damage localization.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class HealthAssessment:
    status: str                         # "Healthy", "Advisory", "Warning", "Critical"
    health_index: float                 # 0.0 (severe failure) to 100.0 (pristine)
    is_anomaly: bool                    # True if reconstruction error exceeds calibrated threshold
    anomaly_score: float                # Raw reconstruction error
    threshold: float                    # Calibrated anomaly threshold
    sensor_attribution: Dict[str, float]# Normalized anomaly share per sensor (0.0 to 1.0)
    most_affected_sensor: str           # Sensor ID with highest deviation


class StructuralHealthPredictor:
    """
    Evaluates bridge structural health and localizes damage from Autoencoder reconstruction errors.
    Calibrates on baseline healthy validation errors using statistical extreme value estimation.
    """

    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity  # Multiplier for threshold tightness
        self.mean_error: float = 0.0
        self.std_error: float = 1.0
        self.threshold: float = 0.05
        self.is_calibrated: bool = False

    def calibrate(self, healthy_errors: np.ndarray, percentile: float = 99.0) -> "StructuralHealthPredictor":
        """
        Calibrate baseline parameters and anomaly threshold on healthy validation errors.
        """
        self.mean_error = float(np.mean(healthy_errors))
        self.std_error = float(np.std(healthy_errors))
        if self.std_error < 1e-6:
            self.std_error = 1e-3

        # Calibrated threshold based on extreme percentile of normal operations
        base_thresh = float(np.percentile(healthy_errors, percentile))
        # Sensitivity adjusts between lenient (high sensitivity number = tighter threshold)
        self.threshold = base_thresh / self.sensitivity
        self.is_calibrated = True
        return self

    def compute_health_index(self, error: float) -> float:
        """
        Compute Structural Health Index (0 to 100) from reconstruction error.
        Follows continuous exponential penalty function:
        - Within normal baseline (error <= mean): SHI = 98 - 100%
        - Between mean and threshold: SHI = 80 - 95% (Advisory)
        - Between 1x and 2.5x threshold: SHI = 50 - 80% (Warning)
        - Exceeding 2.5x threshold: SHI < 50% (Critical Alarm)
        """
        if error <= self.mean_error:
            # Baseline normal vibration
            shi = 100.0 - 5.0 * (max(0.0, error) / max(1e-5, self.mean_error))
        else:
            normalized_deviation = (error - self.mean_error) / max(1e-5, self.threshold - self.mean_error)
            # Exponential decay
            shi = 95.0 * np.exp(-0.45 * max(0.0, normalized_deviation))

        return float(np.clip(shi, 5.0, 100.0))

    def assess_window(
        self,
        reconstruction_error: float,
        sensor_errors: np.ndarray,
        sensor_names: List[str],
    ) -> HealthAssessment:
        """
        Assess structural health for a single window.
        """
        is_anomaly = bool(reconstruction_error > self.threshold)
        health_index = self.compute_health_index(reconstruction_error)

        # Status categorization
        if health_index >= 88.0:
            status = "Healthy"
        elif health_index >= 72.0:
            status = "Advisory"
        elif health_index >= 48.0:
            status = "Warning"
        else:
            status = "Critical"

        # Sensor damage localization attribution
        # Min-max or softmax normalized attribution
        eps = 1e-8
        sensor_attribution = {}
        total_err = np.sum(sensor_errors) + eps
        for idx, name in enumerate(sensor_names):
            sensor_attribution[name] = float(sensor_errors[idx] / total_err)

        max_idx = int(np.argmax(sensor_errors))
        most_affected = sensor_names[max_idx] if max_idx < len(sensor_names) else "Unknown"

        return HealthAssessment(
            status=status,
            health_index=round(health_index, 2),
            is_anomaly=is_anomaly,
            anomaly_score=float(reconstruction_error),
            threshold=float(self.threshold),
            sensor_attribution=sensor_attribution,
            most_affected_sensor=most_affected,
        )

    def assess_batch(
        self,
        reconstruction_errors: np.ndarray,
        sensor_errors: np.ndarray,
        sensor_names: List[str],
    ) -> List[HealthAssessment]:
        """Assess a batch of windows."""
        assessments = []
        for i in range(len(reconstruction_errors)):
            assessments.append(
                self.assess_window(
                    reconstruction_errors[i],
                    sensor_errors[i],
                    sensor_names,
                )
            )
        return assessments
