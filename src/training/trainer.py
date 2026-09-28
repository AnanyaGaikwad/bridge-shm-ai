"""
Training engine for Conv1D-BiLSTM Autoencoder on bridge vibration data.
Supports Apple Silicon MPS acceleration, checkpointing, and validation monitoring.
"""

from dataclasses import dataclass, field
import os
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..models.conv1d_bilstm_autoencoder import Conv1DBiLSTMAutoencoder
from ..models.health_index import StructuralHealthPredictor


def get_optimal_device() -> torch.device:
    """Select MPS for Apple Silicon GPU, CUDA for Nvidia, or CPU."""
    if torch.backends.mps.is_available():
        try:
            # Quick check if MPS is operational
            _ = torch.zeros(1).to("mps")
            return torch.device("mps")
        except Exception:
            return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainingResult:
    best_val_loss: float
    train_losses: List[float]
    val_losses: List[float]
    epochs_trained: int
    checkpoint_path: str


class ModelTrainer:
    """
    Manages end-to-end unsupervised training of the Autoencoder on baseline healthy vibrations.
    """

    def __init__(
        self,
        model: Conv1DBiLSTMAutoencoder,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.device = device or get_optimal_device()
        self.model = model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        # Huber / Smooth L1 loss provides robustness against vehicle impact spikes
        self.criterion = nn.SmoothL1Loss(beta=0.05)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 25,
        patience: int = 7,
        checkpoint_dir: str = "checkpoints",
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> TrainingResult:
        """
        Train the model with early stopping and Cosine Annealing learning rate.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "best_bridge_shm_model.pt")

        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        train_losses: List[float] = []
        val_losses: List[float] = []

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # ----------------- Training -----------------
            self.model.train()
            running_train_loss = 0.0
            num_batches = 0

            for batch_x, _, _ in train_loader:
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(batch_x)
                reconstruction = outputs["reconstruction"]

                loss = self.criterion(reconstruction, batch_x)
                loss.backward()
                # Clip gradients to avoid exploding RNN gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()

                running_train_loss += loss.item()
                num_batches += 1

            epoch_train_loss = running_train_loss / max(1, num_batches)
            train_losses.append(epoch_train_loss)

            # ----------------- Validation -----------------
            epoch_val_loss = epoch_train_loss
            if val_loader is not None:
                self.model.eval()
                running_val_loss = 0.0
                num_val_batches = 0

                with torch.no_grad():
                    for batch_x, _, _ in val_loader:
                        batch_x = batch_x.to(self.device)
                        outputs = self.model(batch_x)
                        reconstruction = outputs["reconstruction"]
                        v_loss = self.criterion(reconstruction, batch_x)
                        running_val_loss += v_loss.item()
                        num_val_batches += 1

                epoch_val_loss = running_val_loss / max(1, num_val_batches)
                val_losses.append(epoch_val_loss)

                # Check for best validation score
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_val_loss": best_val_loss,
                        },
                        checkpoint_path,
                    )
                else:
                    patience_counter += 1
            else:
                val_losses.append(epoch_train_loss)
                if epoch_train_loss < best_val_loss:
                    best_val_loss = epoch_train_loss
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "best_val_loss": best_val_loss,
                        },
                        checkpoint_path,
                    )

            scheduler.step()

            if progress_callback:
                progress_callback(epoch, epochs, epoch_train_loss, epoch_val_loss)

            # Early stopping check
            if patience_counter >= patience:
                break

        # Load best model weights if saved
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])

        return TrainingResult(
            best_val_loss=best_val_loss,
            train_losses=train_losses,
            val_losses=val_losses,
            epochs_trained=len(train_losses),
            checkpoint_path=checkpoint_path,
        )

    def extract_reconstruction_errors(
        self,
        loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute reconstruction errors for all windows in a loader.
        Returns:
            window_errors: Shape [N]
            sensor_errors: Shape [N, num_sensors]
        """
        self.model.eval()
        all_win_err = []
        all_sens_err = []

        with torch.no_grad():
            for batch_x, _, _ in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                all_win_err.append(outputs["reconstruction_error"].cpu().numpy())
                all_sens_err.append(outputs["sensor_errors"].cpu().numpy())

        return np.concatenate(all_win_err, axis=0), np.concatenate(all_sens_err, axis=0)
