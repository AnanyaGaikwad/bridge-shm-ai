"""
End-to-end Pipeline for Bridge Structural Health Monitoring.
Connects simulation, preprocessing, PyTorch model, and health prediction.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from .data.preprocessing import (
    BridgeFeatureScaler,
    butter_bandpass_filter,
    create_sliding_windows,
)
from .models.conv1d_bilstm_autoencoder import Conv1DBiLSTMAutoencoder
from .models.health_index import HealthAssessment, StructuralHealthPredictor
from .simulation.bridge_physics import BridgeDamageScenario, BridgeSimulator
from .training.trainer import ModelTrainer, get_optimal_device


class BridgeSHMPipeline:
    """
    Unified interface for bridge vibration anomaly detection and structural health scoring.
    """

    def __init__(
        self,
        num_sensors: int = 8,
        window_size: int = 256,
        step_size: int = 64,
        sampling_rate: float = 100.0,
        device: Optional[torch.device] = None,
    ):
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.step_size = step_size
        self.fs = sampling_rate
        self.device = device or get_optimal_device()

        self.model = Conv1DBiLSTMAutoencoder(
            num_sensors=num_sensors,
            sequence_length=window_size,
        ).to(self.device)

        self.scaler = BridgeFeatureScaler()
        self.predictor = StructuralHealthPredictor()
        self.simulator = BridgeSimulator(sampling_rate=sampling_rate)
        self.sensor_names = [s.id for s in self.simulator.sensors]
        self.is_trained: bool = False

    def train_on_baseline(
        self,
        duration_s: float = 180.0,
        epochs: int = 20,
        batch_size: int = 32,
        checkpoint_dir: str = "checkpoints",
        progress_callback=None,
    ) -> Dict[str, float]:
        """
        Generate healthy bridge baseline vibration and train the Autoencoder.
        """
        # 1. Simulate healthy operations across temperature variations (15C to 30C)
        all_windows = []
        for temp in [15.0, 22.0, 30.0]:
            sim = self.simulator.simulate(
                duration_s=duration_s / 3.0,
                traffic_intensity=1.0,
                ambient_temp_c=temp,
                damage_scenario=BridgeDamageScenario(damage_type="none"),
                random_seed=int(temp * 10),
            )
            # Bandpass filter
            filt_acc = butter_bandpass_filter(sim["accelerations"], fs=self.fs)
            wins, _, _ = create_sliding_windows(
                filt_acc,
                window_size=self.window_size,
                step_size=self.step_size,
            )
            all_windows.append(wins)

        combined_windows = np.concatenate(all_windows, axis=0)

        # 2. Fit feature scaler on healthy baseline
        self.scaler.fit(combined_windows)
        scaled_windows = self.scaler.transform(combined_windows)

        # 3. Create DataLoaders
        from .data.dataset import create_dataloaders
        train_loader, val_loader = create_dataloaders(
            scaled_windows,
            batch_size=batch_size,
            train_ratio=0.85,
        )

        # 4. Train Model
        trainer = ModelTrainer(self.model, device=self.device)
        train_res = trainer.train(
            train_loader,
            val_loader,
            epochs=epochs,
            checkpoint_dir=checkpoint_dir,
            progress_callback=progress_callback,
        )

        # 5. Calibrate dynamic threshold on validation reconstruction errors
        val_win_err, _ = trainer.extract_reconstruction_errors(val_loader)
        self.predictor.calibrate(val_win_err, percentile=98.5)
        self.is_trained = True

        return {
            "best_val_loss": train_res.best_val_loss,
            "calibrated_threshold": self.predictor.threshold,
            "baseline_mean_error": self.predictor.mean_error,
            "epochs_trained": train_res.epochs_trained,
        }

    def process_continuous_signal(
        self,
        accelerations: np.ndarray,
        sensor_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[HealthAssessment], np.ndarray]:
        """
        Process continuous multi-sensor acceleration records.
        Returns:
            window_times: time center for each window
            assessments: list of HealthAssessment objects
            reconstructions: reconstructed windows
        """
        if not self.is_trained:
            raise RuntimeError("Pipeline model is not trained yet. Run train_on_baseline() or load_checkpoint().")

        names = sensor_names or self.sensor_names
        # 1. Bandpass filter
        filt_acc = butter_bandpass_filter(accelerations, fs=self.fs)

        # 2. Windowing
        windows, _, _ = create_sliding_windows(
            filt_acc,
            window_size=self.window_size,
            step_size=self.step_size,
        )

        # 3. Scale
        scaled_windows = self.scaler.transform(windows)

        # 4. Neural Network Inference
        self.model.eval()
        all_win_err = []
        all_sens_err = []
        all_rec = []

        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(scaled_windows), batch_size):
                b_x = torch.tensor(
                    scaled_windows[i : i + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                out = self.model(b_x)
                all_win_err.append(out["reconstruction_error"].cpu().numpy())
                all_sens_err.append(out["sensor_errors"].cpu().numpy())
                all_rec.append(out["reconstruction"].cpu().numpy())

        win_errors = np.concatenate(all_win_err, axis=0)
        sens_errors = np.concatenate(all_sens_err, axis=0)
        rec_scaled = np.concatenate(all_rec, axis=0)
        rec_unscaled = self.scaler.inverse_transform(rec_scaled)

        # 5. Health Assessment
        assessments = self.predictor.assess_batch(win_errors, sens_errors, names)

        # Calculate time centers for windows
        num_windows = len(windows)
        window_centers = np.array([
            (i * self.step_size + self.window_size / 2.0) / self.fs
            for i in range(num_windows)
        ])

        return window_centers, assessments, rec_unscaled

    def save_checkpoint(self, path: str):
        """Save full pipeline bundle (model, scaler, predictor, configs)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        bundle = {
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler.mean,
            "scaler_std": self.scaler.std,
            "predictor_threshold": self.predictor.threshold,
            "predictor_base_threshold": self.predictor.base_threshold,
            "predictor_mean": self.predictor.mean_error,
            "predictor_std": self.predictor.std_error,
            "num_sensors": self.num_sensors,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "fs": self.fs,
            "sensor_names": self.sensor_names,
        }
        torch.save(bundle, path)

    def load_checkpoint(self, path: str):
        """Load pipeline bundle."""
        bundle = torch.load(path, map_location=self.device, weights_only=False)
        self.num_sensors = bundle["num_sensors"]
        self.window_size = bundle["window_size"]
        self.step_size = bundle["step_size"]
        self.fs = bundle["fs"]
        self.sensor_names = bundle["sensor_names"]

        self.model = Conv1DBiLSTMAutoencoder(
            num_sensors=self.num_sensors,
            sequence_length=self.window_size,
        ).to(self.device)
        self.model.load_state_dict(bundle["model_state"])

        self.scaler.mean = bundle["scaler_mean"]
        self.scaler.std = bundle["scaler_std"]

        self.predictor.threshold = bundle["predictor_threshold"]
        self.predictor.base_threshold = bundle.get(
        "predictor_base_threshold",
        self.predictor.threshold,
        )
        self.predictor.mean_error = bundle["predictor_mean"]
        self.predictor.std_error = bundle["predictor_std"]
        self.predictor.is_calibrated = True
        self.is_trained = True
