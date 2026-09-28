"""
Quickstart script:
1. Simulates baseline bridge vibrations
2. Trains the Conv1D-BiLSTM Autoencoder
3. Calibrates anomaly thresholds & Structural Health Index
4. Tests across multiple damage scenarios (healthy, minor crack, severe crack)
5. Exports trained checkpoint and sample benchmark CSVs for dashboard demo
"""

import os
import sys
import numpy as np
import pandas as pd

from src.pipeline import BridgeSHMPipeline
from src.simulation.bridge_physics import BridgeDamageScenario, BridgeSimulator
from src.training.evaluator import ModelEvaluator
from src.data.preprocessing import butter_bandpass_filter, create_sliding_windows


def main():
    print("=" * 70)
    print("  BRIDGE STRUCTURAL HEALTH MONITORING (SHM) - DEEP LEARNING")
    print("=" * 70)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    pipeline_ckpt = os.path.join(checkpoint_dir, "bridge_shm_pipeline.pt")

    pipeline = BridgeSHMPipeline(
        num_sensors=8,
        window_size=256,
        step_size=64,
        sampling_rate=100.0,
    )
    print(f"[1/4] Initialized pipeline on device: {pipeline.device}")

    # Check if checkpoint exists or train new model
    if os.path.exists(pipeline_ckpt):
        print(f"[2/4] Found existing checkpoint at {pipeline_ckpt}. Loading...")
        pipeline.load_checkpoint(pipeline_ckpt)
    else:
        print("[2/4] Simulating healthy baseline vibration & training Autoencoder...")
        def progress(epoch, total, t_loss, v_loss):
            if epoch % 3 == 0 or epoch == total:
                print(f"      Epoch [{epoch:02d}/{total:02d}] - Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f}")

        train_stats = pipeline.train_on_baseline(
            duration_s=180.0,
            epochs=15,
            batch_size=32,
            checkpoint_dir=checkpoint_dir,
            progress_callback=progress,
        )
        print(f"      Training complete! Best Val Loss: {train_stats['best_val_loss']:.5f}")
        print(f"      Calibrated Anomaly Threshold: {train_stats['calibrated_threshold']:.5f}")
        pipeline.save_checkpoint(pipeline_ckpt)
        print(f"      Saved model bundle to {pipeline_ckpt}")

    # 3. Simulate and save sample benchmark datasets
    print("[3/4] Generating benchmark evaluation scenarios...")
    sim = BridgeSimulator(sampling_rate=100.0)

    # Scenario A: Healthy operational day (traffic + temperature drift)
    sim_healthy = sim.simulate(
        duration_s=60.0,
        traffic_intensity=1.2,
        ambient_temp_c=24.0,
        damage_scenario=BridgeDamageScenario(damage_type="none"),
        random_seed=101,
    )
    df_healthy = sim.generate_dataframe(sim_healthy)
    df_healthy.to_csv("data/sample_healthy.csv", index=False)
    print("      Saved 'data/sample_healthy.csv'")

    # Scenario B: Structural crack at Span 2 Midspan (location = 110m, severity = 0.35)
    sim_damaged = sim.simulate(
        duration_s=60.0,
        traffic_intensity=1.2,
        ambient_temp_c=24.0,
        damage_scenario=BridgeDamageScenario(
            damage_type="crack",
            location_m=110.0,
            severity=0.35,
            start_time_s=15.0, # damage occurs after 15s
        ),
        random_seed=202,
    )
    df_damaged = sim.generate_dataframe(sim_damaged)
    df_damaged.to_csv("data/sample_damaged.csv", index=False)
    print("      Saved 'data/sample_damaged.csv'")

    # 4. Run Model Evaluation
    print("[4/4] Evaluating Deep Learning Anomaly Detection & Health Scoring...")
    
    # Process healthy test run
    times_h, assess_h, _ = pipeline.process_continuous_signal(sim_healthy["accelerations"])
    avg_shi_h = np.mean([a.health_index for a in assess_h])
    anom_count_h = sum(1 for a in assess_h if a.is_anomaly)
    print(f"\n  --- Healthy Scenario Results ---")
    print(f"  Average Structural Health Index (SHI): {avg_shi_h:.1f}% / 100%")
    print(f"  Total Windows: {len(assess_h)} | Flagged Anomalies: {anom_count_h} (False Alarm Rate: {anom_count_h/len(assess_h):.1%})")

    # Process damaged test run
    times_d, assess_d, _ = pipeline.process_continuous_signal(sim_damaged["accelerations"])
    # Windows after damage injection (time > 15s)
    damaged_windows = [a for t, a in zip(times_d, assess_d) if t >= 16.0]
    avg_shi_d = np.mean([a.health_index for a in damaged_windows])
    detected_d = sum(1 for a in damaged_windows if a.is_anomaly)
    
    print(f"\n  --- Damaged Scenario Results (Crack @ Midspan 2) ---")
    print(f"  Post-damage Structural Health Index: {avg_shi_d:.1f}% (Status: {damaged_windows[-1].status})")
    print(f"  Detection Sensitivity (Recall): {detected_d}/{len(damaged_windows)} ({detected_d/len(damaged_windows):.1%})")
    print(f"  Most Damaged Sensor Identified: {damaged_windows[-1].most_affected_sensor}")
    print(f"  Attribution to S5_S2M (Sensor at 110m): {damaged_windows[-1].sensor_attribution.get('S5_S2M', 0.0):.1%}")

    print("\n" + "=" * 70)
    print("  PIPELINE READY! Launch Streamlit dashboard with:")
    print("  streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
