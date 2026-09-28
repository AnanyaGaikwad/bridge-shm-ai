"""
Simulation modules for bridge physical dynamics and structural damage injection.
"""
from .bridge_physics import BridgeSimulator, BridgeDamageScenario, SensorConfig

__all__ = ["BridgeSimulator", "BridgeDamageScenario", "SensorConfig"]
