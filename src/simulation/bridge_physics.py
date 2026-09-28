"""
Physical dynamics simulator for continuous multi-span highway/rail bridges.
Implements multi-degree-of-freedom modal dynamics with moving vehicular loads,
diurnal environmental thermal effects, wind turbulence, and structural damage injection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal


@dataclass
class SensorConfig:
    id: str
    location_m: float
    span_index: int
    description: str


@dataclass
class BridgeDamageScenario:
    damage_type: str = "none"  # "none", "stiffness_loss", "crack", "bearing_stiffness"
    location_m: float = 110.0  # Location along the bridge deck
    severity: float = 0.0      # 0.0 (pristine) to 0.6 (severe degradation)
    start_time_s: float = 0.0  # When damage initiates (seconds)


class BridgeSimulator:
    """
    Simulates dynamic vibration response of a 3-span continuous girder bridge.
    Bridge geometry:
        Total Length L = 220m
        Span 1 = 60m (0m - 60m)
        Span 2 = 100m (60m - 160m, main span)
        Span 3 = 60m (160m - 220m)
    """

    def __init__(
        self,
        total_length: float = 220.0,
        span_lengths: Tuple[float, float, float] = (60.0, 100.0, 60.0),
        sampling_rate: float = 100.0,
        num_modes: int = 5,
        nominal_frequencies: Optional[List[float]] = None,
        damping_ratios: Optional[List[float]] = None,
    ):
        self.L = total_length
        self.span_lengths = span_lengths
        self.fs = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.num_modes = num_modes

        # Nominal healthy modal natural frequencies (Hz)
        # Mode 1: 1st vertical bending (2.45 Hz)
        # Mode 2: 1st asymmetric bending (4.75 Hz)
        # Mode 3: 2nd vertical bending (7.80 Hz)
        # Mode 4: 1st torsional/higher vertical (11.50 Hz)
        # Mode 5: Higher mode (16.20 Hz)
        self.nominal_freqs = (
            np.array(nominal_frequencies, dtype=np.float64)
            if nominal_frequencies
            else np.array([2.45, 4.75, 7.80, 11.50, 16.20])
        )

        # Modal damping ratios (typical concrete/steel bridges ~ 1.5% - 2.5%)
        self.damping_ratios = (
            np.array(damping_ratios, dtype=np.float64)
            if damping_ratios
            else np.array([0.018, 0.020, 0.022, 0.025, 0.028])
        )

        # 8 Sensor layout across spans
        self.sensors: List[SensorConfig] = [
            SensorConfig("S1_S1Q", location_m=25.0, span_index=0, description="Span 1 Quarter-point"),
            SensorConfig("S2_S1M", location_m=45.0, span_index=0, description="Span 1 Midspan"),
            SensorConfig("S3_P1",  location_m=60.0, span_index=0, description="Pier 1 Support"),
            SensorConfig("S4_S2Q", location_m=85.0, span_index=1, description="Span 2 Quarter-point"),
            SensorConfig("S5_S2M", location_m=110.0, span_index=1, description="Span 2 Midspan (Critical)"),
            SensorConfig("S6_S2Q3", location_m=135.0, span_index=1, description="Span 2 Three-Quarter"),
            SensorConfig("S7_P2",  location_m=160.0, span_index=2, description="Pier 2 Support"),
            SensorConfig("S8_S3M", location_m=190.0, span_index=2, description="Span 3 Midspan"),
        ]

    def _mode_shape(self, n: int, x: np.ndarray) -> np.ndarray:
        """
        Analytical approximate mode shapes for continuous multi-span beam.
        Mode n is normalized such that max|phi_n| = 1.0.
        """
        # Multi-span continuous beam modal approximation
        x_norm = np.clip(x / self.L, 0.0, 1.0)
        if n == 0:
            # Fundamental vertical bending (peaks at central span)
            phi = np.sin(np.pi * x_norm) * (0.6 + 0.4 * np.sin(np.pi * x_norm))
        elif n == 1:
            # Asymmetric bending
            phi = np.sin(2 * np.pi * x_norm)
        elif n == 2:
            # 2nd symmetric vertical
            phi = np.sin(3 * np.pi * x_norm)
        elif n == 3:
            # Asymmetric higher mode
            phi = np.sin(4 * np.pi * x_norm)
        else:
            phi = np.sin((n + 1) * np.pi * x_norm)

        # Enforce zero boundary conditions at abutments and piers
        pier1_norm = self.span_lengths[0] / self.L
        pier2_norm = (self.span_lengths[0] + self.span_lengths[1]) / self.L
        attenuation = (1.0 - 0.7 * np.exp(-((x_norm - pier1_norm) ** 2) / 0.001)) * \
                      (1.0 - 0.7 * np.exp(-((x_norm - pier2_norm) ** 2) / 0.001))
        phi = phi * attenuation
        max_val = np.max(np.abs(phi))
        if max_val > 1e-6:
            phi = phi / max_val
        return phi

    def simulate(
        self,
        duration_s: float = 60.0,
        traffic_intensity: float = 1.0,     # Multiplier for vehicle arrival rate
        mean_vehicle_speed_kmh: float = 65.0, # Average traffic speed
        ambient_temp_c: float = 22.0,       # Temperature affecting stiffness
        wind_turbulence_level: float = 0.03, # Wind excitation strength
        noise_level_g: float = 0.004,       # Accelerometer measurement noise in g
        damage_scenario: Optional[BridgeDamageScenario] = None,
        random_seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate multi-sensor vibration acceleration signals.
        Returns dictionary with:
            'time': 1D array of time steps
            'accelerations': 2D array of shape [num_sensors, num_timesteps] (in m/s^2)
            'sensor_names': list of sensor ID strings
            'damage_labels': binary anomaly array [num_timesteps] (0: normal, 1: damaged)
            'health_index': true structural health index [num_timesteps] (100: pristine, 0: failed)
            'modal_freqs': active natural frequencies under simulation conditions
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        num_steps = int(duration_s * self.fs)
        time = np.linspace(0, duration_s, num_steps, endpoint=False)
        num_sensors = len(self.sensors)
        sensor_locs = np.array([s.location_m for s in self.sensors])

        # Temperature effect on Young's modulus E: ~ -0.15% per deg C deviation from 20C
        temp_factor = 1.0 - 0.0015 * (ambient_temp_c - 20.0)
        base_freqs = self.nominal_freqs * np.sqrt(max(0.5, temp_factor))

        # Setup damage parameters
        damage = damage_scenario or BridgeDamageScenario()
        active_freqs = base_freqs.copy()

        # Damage localized stiffness reduction impact on natural frequencies
        if damage.damage_type in ("stiffness_loss", "crack") and damage.severity > 0:
            # Mode shape sensitivity at damage location
            sensitivities = np.array([
                float(self._mode_shape(n, np.array([damage.location_m]))[0] ** 2)
                for n in range(self.num_modes)
            ])
            # Frequency reduction factor: delta_omega / omega ~= 0.5 * delta_k / k * phi^2
            freq_reduction = 0.5 * damage.severity * sensitivities
            freq_factor = np.clip(1.0 - freq_reduction, 0.4, 1.0)
        elif damage.damage_type == "bearing_stiffness" and damage.severity > 0:
            # Stiffened/frozen bearing increases mode 1/2 frequency slightly or causes restraint
            freq_factor = 1.0 + 0.1 * damage.severity * np.ones(self.num_modes)
        else:
            freq_factor = np.ones(self.num_modes)

        # Modal coordinates q_n(t), velocity dq_n(t), acceleration ddq_n(t)
        # Using discrete state-space / Newmark-beta integration
        accelerations = np.zeros((num_sensors, num_steps), dtype=np.float64)

        # Generate traffic moving loads
        # Poisson vehicle arrivals
        arrival_rate = 0.35 * traffic_intensity # vehicles per second
        total_expected_vehicles = max(2, int(arrival_rate * (duration_s + self.L / 10.0)))
        inter_arrival_times = np.random.exponential(1.0 / arrival_rate, size=total_expected_vehicles)
        entry_times = np.cumsum(inter_arrival_times) - (self.L / 18.0) # some already on bridge

        # Vehicle masses in kg (cars: 1500kg, trucks: 15000-35000kg)
        is_truck = np.random.rand(total_expected_vehicles) < 0.28
        vehicle_masses = np.where(
            is_truck,
            np.random.uniform(18000, 38000, size=total_expected_vehicles),
            np.random.uniform(1200, 2400, size=total_expected_vehicles)
        )
        speeds_ms = (mean_vehicle_speed_kmh + np.random.normal(0, 7.0, size=total_expected_vehicles)) * (1000.0 / 3600.0)
        speeds_ms = np.clip(speeds_ms, 8.0, 35.0)

        # Modal response calculation
        g = 9.81
        omega_base = 2.0 * np.pi * base_freqs
        omega_dmg = 2.0 * np.pi * (base_freqs * freq_factor)

        # Precompute mode shapes at sensor locations: shape [num_modes, num_sensors]
        phi_sensors_healthy = np.zeros((self.num_modes, num_sensors))
        phi_sensors_damaged = np.zeros((self.num_modes, num_sensors))
        for m in range(self.num_modes):
            phi_sensors_healthy[m, :] = self._mode_shape(m, sensor_locs)
            phi_sensors_damaged[m, :] = phi_sensors_healthy[m, :].copy()
            # Local perturbation if damage is present
            if damage.severity > 0:
                dist = np.abs(sensor_locs - damage.location_m)
                local_boost = damage.severity * np.exp(-((dist / 18.0) ** 2))
                phi_sensors_damaged[m, :] += local_boost * (-0.4 if m % 2 == 0 else 0.4)

        # Modal force time-series F_m[t]
        modal_forces = np.zeros((self.num_modes, num_steps), dtype=np.float64)

        for v_idx in range(total_expected_vehicles):
            t_entry = entry_times[v_idx]
            v_speed = speeds_ms[v_idx]
            t_exit = t_entry + (self.L / v_speed)
            
            # Find time indices where vehicle is on bridge
            idx_start = max(0, int(np.floor(t_entry * self.fs)))
            idx_end = min(num_steps, int(np.ceil(t_exit * self.fs)))
            if idx_end > idx_start:
                t_sub = time[idx_start:idx_end]
                x_veh = v_speed * (t_sub - t_entry)
                load_n = vehicle_masses[v_idx] * g * 1e-4 # scale factor for modal mass

                # Dynamic vehicle tire/suspension bounce excitation
                bounce = 1.0 + 0.20 * np.sin(2.0 * np.pi * 3.5 * (t_sub - t_entry))
                for m in range(self.num_modes):
                    phi_v = self._mode_shape(m, x_veh)
                    modal_forces[m, idx_start:idx_end] += load_n * bounce * phi_v

        # Add ambient wind and micro-tremor forces (pink/white noise)
        for m in range(self.num_modes):
            wind_noise = np.random.normal(0, wind_turbulence_level, size=num_steps)
            modal_forces[m, :] += wind_noise

        # Solve SDOF for each mode using discrete digital filter (IIR filter of 2nd order harmonic oscillator)
        ddq = np.zeros((self.num_modes, num_steps), dtype=np.float64)
        ddq_norm = np.zeros((self.num_modes, num_steps), dtype=np.float64)
        damage_active_mask = (time >= damage.start_time_s) & (damage.severity > 0)

        for m in range(self.num_modes):
            # Normal state transfer function
            w_norm = omega_base[m]
            zeta = self.damping_ratios[m]
            
            # SDOF continuous system: H(s) = s^2 / (s^2 + 2*zeta*w*s + w^2)
            # Bilinear transform discretization
            num = [1.0, 0.0, 0.0]
            den = [1.0, 2.0 * zeta * w_norm, w_norm ** 2]
            b_norm, a_norm = signal.bilinear(num, den, fs=self.fs)
            
            # Damaged state transfer function
            w_dam = omega_dmg[m]
            den_dam = [1.0, 2.0 * zeta * w_dam, w_dam ** 2]
            b_dam, a_dam = signal.bilinear(num, den_dam, fs=self.fs)

            # Filter forces to obtain modal acceleration
            ddq_norm[m, :] = signal.lfilter(b_norm, a_norm, modal_forces[m, :])
            if np.any(damage_active_mask):
                ddq_dam = signal.lfilter(b_dam, a_dam, modal_forces[m, :])
                ddq[m, :] = np.where(damage_active_mask, ddq_dam, ddq_norm[m, :])
                # If breathing crack, add non-linear harmonic distortion
                if damage.damage_type == "crack" and damage.severity > 0:
                    crack_harmonics = np.sin(4.0 * np.pi * base_freqs[m] * time) * (damage.severity * 0.9 * np.abs(ddq[m, :]))
                    ddq[m, :] = np.where(damage_active_mask, ddq[m, :] + crack_harmonics, ddq[m, :])
            else:
                ddq[m, :] = ddq_norm[m, :]

        # Sensor amplification factor near damage zone (strain concentration)
        amplification = np.ones(num_sensors)
        if damage.severity > 0:
            dist = np.abs(sensor_locs - damage.location_m)
            amplification += damage.severity * 2.2 * np.exp(-((dist / 22.0) ** 2))

        # Reconstruct physical accelerations at sensor stations: a(x_s, t) = sum_m phi_m(x_s) * ddq_m(t)
        for s_idx in range(num_sensors):
            acc_healthy = np.zeros(num_steps)
            acc_damaged = np.zeros(num_steps)
            for m in range(self.num_modes):
                acc_healthy += phi_sensors_healthy[m, s_idx] * ddq_norm[m, :]
                acc_damaged += phi_sensors_damaged[m, s_idx] * ddq[m, :] * amplification[s_idx]
            
            acc_s = np.where(damage_active_mask, acc_damaged, acc_healthy)
            # Add sensor electronics noise (Gaussian + small high freq drift)
            sensor_noise = np.random.normal(0, noise_level_g * 9.81, size=num_steps)
            accelerations[s_idx, :] = acc_s + sensor_noise

        # True labels and health index (0 to 100)
        damage_labels = np.where(damage_active_mask, 1, 0)
        
        # Structural Health Index: 100 for healthy, declines with damage severity
        # Scale: severity 0.1 -> SHI ~ 88%, severity 0.3 -> SHI ~ 65%, severity 0.5 -> SHI ~ 38%
        if damage.severity > 0:
            shi_val = max(10.0, 100.0 - (damage.severity * 140.0))
            health_index = np.where(damage_active_mask, shi_val, 100.0)
        else:
            health_index = np.full(num_steps, 100.0)

        sensor_names = [s.id for s in self.sensors]

        return {
            "time": time,
            "accelerations": accelerations,
            "sensor_names": sensor_names,
            "damage_labels": damage_labels,
            "health_index": health_index,
            "base_frequencies": base_freqs,
            "active_frequencies": omega_dmg / (2.0 * np.pi) if damage.severity > 0 else base_freqs,
            "sampling_rate": self.fs,
            "damage_scenario": {
                "type": damage.damage_type,
                "location_m": damage.location_m,
                "severity": damage.severity,
                "start_time_s": damage.start_time_s,
            },
        }

    def generate_dataframe(self, sim_result: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Convert simulation results into a clean pandas DataFrame."""
        df_dict = {"time_s": sim_result["time"]}
        for idx, name in enumerate(sim_result["sensor_names"]):
            df_dict[f"acc_{name}_m_s2"] = sim_result["accelerations"][idx]
        df_dict["is_anomaly"] = sim_result["damage_labels"]
        df_dict["health_index"] = sim_result["health_index"]
        return pd.DataFrame(df_dict)
