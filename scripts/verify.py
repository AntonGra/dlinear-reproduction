"""Independently reconstruct test targets and verify saved model predictions.

Only loads the checkpoint created locally by the selected run. Never downloads
weights or data. Writes a new verification report without overwriting files.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from run import ROOT, environment, source_state

os.environ.update(environment())
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(ROOT / "upstream"))
from models.DLinear import Model
from types import SimpleNamespace


def verify(run_id):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", run_id):
        raise ValueError("Invalid run name")
    folder = ROOT / "runs" / run_id
    meta = json.loads((folder / "metadata.json").read_text())
    assert meta["exit_code"] == 0 and meta["sources_unchanged"]
    assert source_state() == meta["source_hashes"]
    data_file = ROOT / "data/ETTh1.csv"
    assert hashlib.sha256(data_file.read_bytes()).hexdigest() == meta["data_sha256"]
    frame = pd.read_csv(data_file)
    assert list(frame.columns) == ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    times = pd.to_datetime(frame["date"])
    assert times.is_unique and times.is_monotonic_increasing
    assert (times.diff().dropna() == pd.Timedelta(hours=1)).all()
    raw = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    assert np.isfinite(raw).all() and len(raw) >= 14400
    # Reconstruct upstream's 12/4/4 x 30-day chronological split independently.
    # Fit normalization on training rows ONLY, then generate rolling test windows.
    train = raw[:8640]
    mean, scale = train.mean(axis=0), train.std(axis=0, ddof=0)
    assert (scale > 0).all()
    scaled = ((raw - mean) / scale).astype(np.float32)
    config = meta["config"]
    assert config["seq_len"] == 336 and config["pred_len"] == 96
    assert config["features"] == "M" and config["num_workers"] == 0
    origins = range(11520, 14400 - 96 + 1)
    targets = np.stack([scaled[i:i+96] for i in origins])
    predictions = list((folder / "results").glob("*/pred.npy"))
    checkpoints = list((folder / "checkpoints").glob("*/checkpoint.pth"))
    assert len(predictions) == len(checkpoints) == 1
    pred = np.load(predictions[0], allow_pickle=False)
    assert pred.shape == targets.shape == (2785, 96, 7)
    assert np.isfinite(pred).all()
    mse = float(np.mean((pred - targets) ** 2))
    mae = float(np.mean(np.abs(pred - targets)))
    log = (folder / "train.log").read_text()
    logged = re.findall(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)", log)
    assert len(logged) == 1
    log_mse, log_mae = map(float, logged[0])
    assert np.isclose(mse, log_mse, rtol=1e-6, atol=1e-8)
    assert np.isclose(mae, log_mae, rtol=1e-6, atol=1e-8)
    assert "Use CPU" in log
    assert "train 8209" in log and "val 2785" in log and "test 2785" in log
    # Check that the locally produced checkpoint really generates the saved
    # forecasts. Match upstream batch size, including the final one-item batch.
    model = Model(SimpleNamespace(seq_len=336, pred_len=96, enc_in=7, individual=False))
    state = torch.load(checkpoints[0], map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    maximum_error = 0.0
    origin_list = list(origins)
    with torch.no_grad():
        for start in range(0, len(origin_list), 32):
            batch = np.stack([scaled[i-336:i] for i in origin_list[start:start+32]])
            actual = model(torch.from_numpy(batch)).numpy()
            expected = pred[start:start+len(batch)]
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    epochs = re.findall(r"Epoch: (\d+), Steps: (\d+) \| Train Loss: ([\d.]+) Vali Loss: ([\d.]+) Test Loss: ([\d.]+)", log)
    assert epochs
    best = min(epochs, key=lambda x: float(x[3]))
    result = {
        "run_id": run_id, "status": "PASS", "mse": mse, "mae": mae,
        "logged_mse": log_mse, "logged_mae": log_mae,
        "prediction_shape": list(pred.shape), "all_test_windows_evaluated": True,
        "test_windows": len(origins), "last_test_batch_size": len(origins) % 32,
        "checkpoint_prediction_max_abs_error": maximum_error,
        "checkpoint_sha256": hashlib.sha256(checkpoints[0].read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions[0].read_bytes()).hexdigest(),
        "dataset_rows": len(frame), "missing_values": int(frame.isna().sum().sum()),
        "train_rows": 8640, "val_rows": 2880, "test_rows": 2880,
        "train_windows": 8209, "val_windows": 2785,
        "epochs_completed": len(epochs), "best_validation_epoch": int(best[0]),
        "elapsed_seconds": meta["elapsed_seconds"], "torch_threads": torch.get_num_threads(),
        "source_integrity": "PASS", "normalization": "training-only, independently reconstructed",
        "metrics_space": "standardized; mean over test windows, forecast horizon, and 7 channels",
    }
    with (folder / "verification.json").open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    verify(parser.parse_args().run_id)
