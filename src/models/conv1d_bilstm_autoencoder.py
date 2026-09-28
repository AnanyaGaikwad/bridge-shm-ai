"""
Hybrid Conv1D - Bidirectional LSTM Autoencoder for multi-channel vibration anomaly detection.
Combines 1D multi-scale temporal convolutions for capturing modal vibration wavelets
with Bidirectional LSTM layers for modeling structural resonance, decay, and inter-sensor dynamics.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


class Conv1DBiLSTMAutoencoder(nn.Module):
    """
    Symmetric Convolutional-Recurrent Autoencoder.
    Input tensor shape: [Batch, num_sensors, sequence_length]
    Output tensor shape: [Batch, num_sensors, sequence_length]
    """

    def __init__(
        self,
        num_sensors: int = 8,
        sequence_length: int = 256,
        conv_channels: Tuple[int, int, int] = (32, 64, 96),
        lstm_hidden: int = 64,
        latent_dim: int = 48,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.seq_len = sequence_length
        self.conv_channels = conv_channels
        self.lstm_hidden = lstm_hidden
        self.latent_dim = latent_dim

        # -----------------------------
        # 1. ENCODER
        # -----------------------------
        # Downsampling factor: 2 * 2 * 2 = 8
        self.enc_conv1 = nn.Sequential(
            nn.Conv1d(num_sensors, conv_channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(conv_channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(conv_channels[1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(conv_channels[2]),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # BiLSTM Encoder
        self.enc_lstm = nn.LSTM(
            input_size=conv_channels[2],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # Latent projection: 2 * lstm_hidden -> latent_dim
        self.to_latent = nn.Linear(lstm_hidden * 2, latent_dim)

        # -----------------------------
        # 2. DECODER
        # -----------------------------
        self.from_latent = nn.Linear(latent_dim, lstm_hidden * 2)
        self.dec_lstm = nn.LSTM(
            input_size=lstm_hidden * 2,
            hidden_size=conv_channels[2] // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Upsampling ConvTranspose layers (factor of 8 back to original length)
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose1d(
                conv_channels[2],
                conv_channels[1],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm1d(conv_channels[1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
        )
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose1d(
                conv_channels[1],
                conv_channels[0],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm1d(conv_channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec_conv3 = nn.ConvTranspose1d(
            conv_channels[0],
            num_sensors,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, C, T]
        Returns:
            latent: [B, T_down, latent_dim]
            bottleneck_summary: [B, latent_dim] (mean pooled)
        """
        # Convolutions
        h = self.enc_conv1(x)
        h = self.enc_conv2(h)
        h = self.enc_conv3(h)  # [B, C3, T/8]

        # Reshape for LSTM: [B, T/8, C3]
        h = h.transpose(1, 2)
        lstm_out, _ = self.enc_lstm(h)  # [B, T/8, 2 * lstm_hidden]
        latent = self.to_latent(lstm_out)  # [B, T/8, latent_dim]
        summary = torch.mean(latent, dim=1)
        return latent, summary

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latent: [B, T_down, latent_dim]
        Returns reconstructed x: [B, C, T]
        """
        h = self.from_latent(latent)
        lstm_out, _ = self.dec_lstm(h)  # [B, T/8, C3]
        
        # Reshape for ConvTranspose: [B, C3, T/8]
        h = lstm_out.transpose(1, 2)
        h = self.dec_conv1(h)
        h = self.dec_conv2(h)
        reconstruction = self.dec_conv3(h)  # [B, C, T]
        return reconstruction

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Returns dictionary containing:
            'reconstruction': Reconstructed vibration signals [B, C, T]
            'latent': Latent sequence representation [B, T/8, latent_dim]
            'summary': Mean pooled latent vector [B, latent_dim]
            'reconstruction_error': Window-level MSE [B]
            'sensor_errors': Sensor-wise MSE [B, C]
        """
        latent, summary = self.encode(x)
        reconstruction = self.decode(latent)

        # Ensure exact length match in case of odd sequence padding
        if reconstruction.shape[-1] != x.shape[-1]:
            reconstruction = reconstruction[:, :, : x.shape[-1]]

        # Compute point-wise squared error: [B, C, T]
        sq_err = (x - reconstruction) ** 2

        # Sensor-wise reconstruction error: [B, C]
        sensor_errors = torch.mean(sq_err, dim=-1)

        # Global window reconstruction error: [B]
        window_error = torch.mean(sq_err, dim=(1, 2))

        return {
            "reconstruction": reconstruction,
            "latent": latent,
            "summary": summary,
            "reconstruction_error": window_error,
            "sensor_errors": sensor_errors,
        }
