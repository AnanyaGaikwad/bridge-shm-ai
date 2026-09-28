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

    # 4. Run Model Evaluation on the two held-out synthetic benchmark runs.
    # These runs are generated after training and are not part of the healthy
    # training/validation windows. The evaluator computes metrics directly
    # from the saved checkpoint rather than relying on dashboard constants.
    print("[4/4] Evaluating Deep Learning Anomaly Detection & Health Scoring...")

    eval_windows = []
    eval_labels = []

    for sim_result in (sim_healthy, sim_damaged):
        filt = butter_bandpass_filter(sim_result["accelerations"], fs=100.0)
        windows, labels, _ = create_sliding_windows(
            filt,
            window_size=pipeline.window_size,
            step_size=pipeline.step_size,
            labels=sim_result["damage_labels"],
            health_index=sim_result["health_index"],
        )
        eval_windows.append(pipeline.scaler.transform(windows))
        eval_labels.append(labels)

    eval_windows = np.concatenate(eval_windows, axis=0)
    eval_labels = np.concatenate(eval_labels, axis=0)

    evaluator = ModelEvaluator(pipeline.model, pipeline.predictor, device=pipeline.device)
    result = evaluator.evaluate(
        eval_windows,
        eval_labels,
        sensor_names=pipeline.sensor_names,
    )

    os.makedirs("artifacts", exist_ok=True)
    import json
    metrics_artifact = {
        "benchmark": "held-out synthetic demo benchmark",
        "checkpoint": pipeline_ckpt,
        "threshold": result.threshold,
        "roc_auc": result.roc_auc,
        "pr_auc": result.pr_auc,
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "false_alarm_rate": result.false_alarm_rate,
        "accuracy": result.accuracy,
        "confusion_matrix": result.confusion_matrix,
        "evaluation_set": {
            "num_windows": int(len(eval_labels)),
            "healthy_windows": int(np.sum(eval_labels == 0)),
            "anomalous_windows": int(np.sum(eval_labels == 1)),
            "healthy_run": "60s, traffic=1.2x, temperature=24C, seed=101",
            "damaged_run": "60s, crack at 110m, severity=0.35, damage starts at 15s, seed=202",
        },
    }
    with open("artifacts/evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_artifact, f, indent=2)

    print("\n  --- Held-Out Synthetic Benchmark ---")
    print(f"  Windows: {len(eval_labels)} ({np.sum(eval_labels == 0)} healthy / {np.sum(eval_labels == 1)} anomalous)")
    print(f"  ROC-AUC: {result.roc_auc:.3f} | PR-AUC: {result.pr_auc:.3f}")
    print(f"  F1: {result.f1_score:.3f} | Recall: {result.recall:.1%} | Precision: {result.precision:.1%}")
    print(f"  False Alarm Rate: {result.false_alarm_rate:.1%}")
    print(f"  Confusion Matrix: {result.confusion_matrix}")
    print("  Saved 'artifacts/evaluation_metrics.json'")

    print("\n" + "=" * 70)
    print("  PIPELINE READY! Launch Streamlit dashboard with:")
    print("  streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
