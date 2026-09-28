"""
PyTorch Dataset and DataLoader wrappers for multi-sensor bridge vibration data,
supporting both simulated dynamics and external CSV files.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class BridgeVibrationDataset(Dataset):
    """
    PyTorch Dataset wrapping sliding window tensors of bridge accelerations.
    X: shape [N, num_sensors, window_size]
    y: shape [N] (anomaly labels, 0 = healthy, 1 = anomalous)
    shi: shape [N] (structural health index, 0.0 to 100.0)
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: Optional[np.ndarray] = None,
        health_index: Optional[np.ndarray] = None,
    ):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = torch.zeros(len(windows), dtype=torch.long)

        if health_index is not None:
            self.health_index = torch.tensor(health_index, dtype=torch.float32)
        else:
            self.health_index = torch.full((len(windows),), 100.0, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx], self.health_index[idx]


def create_dataloaders(
    windows: np.ndarray,
    labels: Optional[np.ndarray] = None,
    health_index: Optional[np.ndarray] = None,
    batch_size: int = 32,
    train_ratio: float = 0.85,
    shuffle_train: bool = True,
    random_seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Split windows into training and validation DataLoaders.
    """
    num_samples = len(windows)
    indices = np.arange(num_samples)

    if train_ratio >= 1.0:
        dataset = BridgeVibrationDataset(windows, labels, health_index)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_train), None

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    split_idx = int(num_samples * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_ds = BridgeVibrationDataset(
        windows[train_indices],
        labels[train_indices] if labels is not None else None,
        health_index[train_indices] if health_index is not None else None,
    )
    val_ds = BridgeVibrationDataset(
        windows[val_indices],
        labels[val_indices] if labels is not None else None,
        health_index[val_indices] if health_index is not None else None,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def load_sensor_csv(
    file_or_path: Union[str, pd.DataFrame],
    time_column: Optional[str] = None,
    exclude_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
    """
    Load external sensor acceleration data from a CSV file or DataFrame.
    Returns:
        accelerations: Shape [num_sensors, num_timesteps]
        sensor_names: List of column names corresponding to sensors
        time_vector: Optional 1D array of time stamps
    """
    if isinstance(file_or_path, pd.DataFrame):
        df = file_or_path
    else:
        df = pd.read_csv(file_or_path)

    exclude = set(exclude_columns or ["is_anomaly", "health_index", "label"])
    time_vec = None

    # Detect time column if not specified
    if time_column and time_column in df.columns:
        time_vec = df[time_column].to_numpy()
        exclude.add(time_column)
    else:
        for candidate in ["time", "time_s", "timestamp", "t"]:
            if candidate in df.columns:
                time_vec = df[candidate].to_numpy()
                exclude.add(candidate)
                break

    sensor_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    if not sensor_cols:
        raise ValueError("No numeric sensor columns found in dataset.")

    acc_matrix = df[sensor_cols].to_numpy().T # Shape: [num_sensors, num_timesteps]
    return acc_matrix, sensor_cols, time_vec
