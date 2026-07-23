"""End-to-end experiment.

Protocol
--------
1. Train an autoencoder on healthy windows from ONE motor load only.
2. Set the anomaly threshold from a held-out healthy split at that same load.
3. Report detection on faults at the training load (the easy case).
4. Report false alarms and detection at the three unseen loads (the honest case).

Fault labels enter the pipeline only at step 3/4, for scoring.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

try:  # works both as `python -m src.main` and as a flat script on Kaggle
    from . import data, evaluate, features, models, train
except ImportError:  # pragma: no cover
    import data, evaluate, features, models, train  # type: ignore

log = logging.getLogger("bearing-ae")


def load_config(path: str | Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def build_windows(runs: list[data.Run], cfg: dict) -> dict[str, np.ndarray]:
    """Featurise every run and stack, keeping provenance for each window."""
    feats, faulty, loads, labels = [], [], [], []
    for run in runs:
        x = features.featurise(run.signal, cfg["features"])
        if x.size == 0:
            continue
        feats.append(x)
        faulty.append(np.full(len(x), not run.is_healthy))
        loads.append(np.full(len(x), run.load))
        labels.append(np.full(len(x), run.label, dtype=object))
    return {
        "x": np.concatenate(feats),
        "faulty": np.concatenate(faulty),
        "load": np.concatenate(loads),
        "label": np.concatenate(labels),
    }


def run(cfg: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train.set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() and cfg["device"] == "auto" else "cpu"
    log.info("device: %s", device)

    runs = data.discover_runs(
        cfg["data"]["roots"],
        channel=cfg["data"]["channel"],
        include=cfg["data"].get("include"),
        exclude=cfg["data"].get("exclude"),
        rate_rules=cfg["data"].get("sample_rate", {}),
        target_rate=cfg["data"].get("target_rate", 12_000),
    )
    if not runs:
        raise SystemExit(
            "No labelled .mat files found. Check config['data']['roots'] -- on "
            "Kaggle this is usually /kaggle/input/<dataset-slug>."
        )
    log.info("\n%s", data.summarise(runs))
    if not any(r.is_healthy for r in runs):
        raise SystemExit("No healthy runs found; cannot train an unsupervised detector.")

    bundle = build_windows(runs, cfg)
    train_load = cfg["experiment"]["train_load"]

    healthy_at_train = (~bundle["faulty"]) & (bundle["load"] == train_load)
    idx = np.flatnonzero(healthy_at_train)
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * cfg["experiment"]["val_fraction"]))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    scaler = features.StandardScaler().fit(bundle["x"][train_idx])
    x_all = scaler.transform(bundle["x"])

    model = models.build_model(cfg, input_dim=x_all.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model %s | %d parameters", cfg["model"]["kind"], n_params)

    history = train.train_autoencoder(
        model, x_all[train_idx], x_all[val_idx], cfg, device=device
    )
    evaluate.plot_history(history, out_dir / "training_curve.png")

    scores = train.reconstruction_error(model, x_all, device=device)
    threshold = evaluate.choose_threshold(
        scores[val_idx], cfg["experiment"]["threshold_quantile"]
    )
    log.info("threshold (healthy val, q=%.3f): %.6g", cfg["experiment"]["threshold_quantile"], threshold)

    results: dict = {
        "config": cfg,
        "n_parameters": n_params,
        "threshold": threshold,
        "epochs_run": len(history["train"]),
    }

    # --- 1. same operating condition as training --------------------------
    same = bundle["load"] == train_load
    held_out = np.ones(len(scores), dtype=bool)
    held_out[train_idx] = False  # never score windows the model trained on
    mask = same & held_out
    results["same_load"] = evaluate.detection_metrics(
        scores[mask], bundle["faulty"][mask], threshold
    )
    results["same_load"]["recall_by_fault"] = evaluate.per_group_recall(
        scores[mask & bundle["faulty"]], bundle["label"][mask & bundle["faulty"]], threshold
    )
    evaluate.plot_score_distributions(
        scores[mask],
        bundle["faulty"][mask],
        threshold,
        out_dir / "scores_train_load.png",
        f"Reconstruction error, {train_load} hp (training condition)",
    )

    # --- 2. unseen operating conditions -----------------------------------
    shifted = (bundle["load"] != train_load) & held_out
    if shifted.any():
        results["shifted_load"] = evaluate.detection_metrics(
            scores[shifted], bundle["faulty"][shifted], threshold
        )
        evaluate.plot_score_distributions(
            scores[shifted],
            bundle["faulty"][shifted],
            threshold,
            out_dir / "scores_shifted_loads.png",
            "Reconstruction error, unseen motor loads",
        )

    fpr_by_load: dict[int, float] = {}
    for load in sorted(set(bundle["load"].tolist())):
        sel = (bundle["load"] == load) & ~bundle["faulty"] & held_out
        if sel.any():
            fpr_by_load[int(load)] = float((scores[sel] > threshold).mean())
    results["false_positive_rate_by_load"] = fpr_by_load
    if len(fpr_by_load) > 1:
        evaluate.plot_fpr_by_load(fpr_by_load, train_load, out_dir / "fpr_by_load.png")

    # --- 3. cheap baseline -------------------------------------------------
    if cfg["experiment"]["run_kurtosis_baseline"]:
        results["kurtosis_baseline"] = _kurtosis_baseline(runs, cfg, bundle)

    with open(out_dir / "metrics.json", "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    torch.save(model.state_dict(), out_dir / "autoencoder.pt")
    log.info("results written to %s", out_dir)
    return results


def _kurtosis_baseline(runs: list[data.Run], cfg: dict, bundle: dict) -> dict:
    """Spectral kurtosis-style scalar detector, same protocol as the AE."""
    kurt, faulty, loads = [], [], []
    for r in runs:
        w = features.window(r.signal, cfg["features"]["window"], cfg["features"]["hop"])
        if w.size == 0:
            continue
        kurt.append(features.rms_baseline(w))
        faulty.append(np.full(len(w), not r.is_healthy))
        loads.append(np.full(len(w), r.load))
    kurt = np.concatenate(kurt)
    faulty = np.concatenate(faulty)
    loads = np.concatenate(loads)

    train_load = cfg["experiment"]["train_load"]
    healthy_train = (~faulty) & (loads == train_load)
    threshold = float(
        np.quantile(kurt[healthy_train], cfg["experiment"]["threshold_quantile"])
    )
    same = loads == train_load
    out = evaluate.detection_metrics(kurt[same], faulty[same], threshold)
    out["threshold"] = threshold
    shifted = loads != train_load
    if shifted.any():
        out["shifted_load"] = evaluate.detection_metrics(
            kurt[shifted], faulty[shifted], threshold
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dense_logspec.yaml")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    cfg = load_config(args.config)
    results = run(cfg, Path(args.out))
    print(json.dumps({k: v for k, v in results.items() if k != "config"}, indent=2, default=str))


if __name__ == "__main__":
    main()
