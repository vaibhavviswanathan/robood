"""
OOD Detection Evaluation & Benchmarking
========================================
Benchmarks detector performance across methods and configurations.

Usage:
    python evaluate_ood.py --mode synthetic --encoder proxy
    python evaluate_ood.py --mode synthetic --encoder proxy --sweep
    python evaluate_ood.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from vla_ood_detector import VLAOODDetector


# ---------------------------------------------------------------------------
# Frame generation (reuse demo sources)
# ---------------------------------------------------------------------------

def _frames_from_source(source, n: int, kind: str) -> list:
    gen = {"in_dist": source.in_dist_frame,
           "mild_ood": source.mild_ood_frame,
           "hard_ood": source.hard_ood_frame}[kind]
    return [gen() for _ in range(n)]


def make_synthetic_frames(n: int, kind: str, rng: np.random.Generator) -> list:
    from vla_ood_demo import SyntheticFrameSource
    return _frames_from_source(SyntheticFrameSource(rng), n, kind)


def make_source(mode: str, args) -> object:
    """Build frame source from CLI args. Returns an object with the 3-method interface."""
    if mode == "synthetic":
        from vla_ood_demo import SyntheticFrameSource
        return SyntheticFrameSource(np.random.default_rng(args.seed))
    elif mode == "lerobot":
        from frame_sources import LeRobotFrameSource
        return LeRobotFrameSource(
            in_dist_repo=args.dataset_repo,
            mild_ood_repo=args.mild_ood_repo,
            max_frames=args.max_frames,
            rng=np.random.default_rng(args.seed),
        )
    elif mode == "robomimic":
        from frame_sources import RobomimicFrameSource
        return RobomimicFrameSource(
            hdf5_path=args.dataset_path,
            max_frames=args.max_frames,
            rng=np.random.default_rng(args.seed),
        )
    elif mode == "disk":
        from frame_sources import DiskFrameSource
        return DiskFrameSource(
            in_dist_dir=args.dataset_path,
            rng=np.random.default_rng(args.seed),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def make_encoder(name: str):
    if name == "proxy":
        from vla_ood_demo import make_proxy_encoder
        return make_proxy_encoder()
    elif name == "dinov2":
        from vla_ood_demo import make_dinov2_encoder
        return make_dinov2_encoder()
    else:
        raise ValueError(f"Unknown encoder: {name}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    detector: VLAOODDetector,
    in_dist_images: list,
    ood_images: list,
) -> dict:
    """Compute AUROC, FPR@TPR=0.95, and latency metrics."""
    from sklearn.metrics import roc_auc_score, roc_curve

    in_scores = []
    ood_scores = []
    latencies = []

    for img in in_dist_images:
        t0 = time.perf_counter()
        r = detector.score(img)
        latencies.append(time.perf_counter() - t0)
        in_scores.append(r.score)

    for img in ood_images:
        t0 = time.perf_counter()
        r = detector.score(img)
        latencies.append(time.perf_counter() - t0)
        ood_scores.append(r.score)

    scores = np.array(in_scores + ood_scores)
    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ood_scores))])

    auroc = roc_auc_score(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    # FPR at TPR >= 0.95
    idx = np.where(tpr >= 0.95)[0]
    fpr_at_95 = float(fpr[idx[0]]) if len(idx) > 0 else 1.0

    latencies_ms = np.array(latencies) * 1000
    p99_latency = float(np.percentile(latencies_ms, 99))
    mean_latency = float(np.mean(latencies_ms))

    # False positive rate at default threshold
    fp_count = sum(1 for s in in_scores if s >= detector._threshold)
    fpr_default = fp_count / max(len(in_scores), 1)

    return {
        "auroc": round(auroc, 4),
        "fpr_at_tpr95": round(fpr_at_95, 4),
        "fpr_default_threshold": round(fpr_default, 4),
        "latency_p99_ms": round(p99_latency, 3),
        "latency_mean_ms": round(mean_latency, 3),
        "threshold": round(detector._threshold, 4),
        "n_in_dist": len(in_dist_images),
        "n_ood": len(ood_images),
        "mean_in_score": round(float(np.mean(in_scores)), 4),
        "mean_ood_score": round(float(np.mean(ood_scores)), 4),
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_single_eval(
    encoder_fn,
    method: str,
    pca_components,
    knn_k: int,
    fit_images: list,
    in_dist_eval: list,
    ood_eval: list,
    verbose: bool = True,
) -> dict:
    detector = VLAOODDetector(
        encoder_fn=encoder_fn,
        method=method,
        knn_k=knn_k,
        threshold_percentile=95.0,
        pca_components=pca_components,
    )
    detector.fit(fit_images, verbose=verbose)
    metrics = compute_metrics(detector, in_dist_eval, ood_eval)
    metrics["method"] = method
    metrics["pca_components"] = pca_components
    metrics["knn_k"] = knn_k
    return metrics


def run_evaluation(args):
    rng = np.random.default_rng(args.seed)
    encoder_fn = make_encoder(args.encoder)

    print("\n" + "=" * 70)
    print("  OOD Detection Evaluation")
    print("=" * 70)
    print(f"  mode={args.mode}  encoder={args.encoder}  seed={args.seed}")
    print()

    # Generate frames
    print(f"[Eval] generating frames ...")
    source = make_source(args.mode, args)
    fit_images = _frames_from_source(source, args.n_fit, "in_dist")
    in_dist_eval = _frames_from_source(source, args.n_eval, "in_dist")
    ood_eval = _frames_from_source(source, args.n_eval, "hard_ood")

    configs = []

    if args.sweep:
        # Sweep over configurations
        for method in ["mahalanobis", "knn"]:
            for pca in [None, 16, 32]:
                knn_k = 5
                if method == "knn":
                    for k in [3, 5, 10]:
                        configs.append((method, pca, k))
                else:
                    configs.append((method, pca, knn_k))
    else:
        # Default comparison: mahalanobis vs knn
        configs = [
            ("mahalanobis", None, 5),
            ("mahalanobis", 32, 5),
            ("knn", None, 5),
            ("knn", 32, 5),
        ]

    results = []
    for method, pca, knn_k in configs:
        label = f"{method} pca={pca} k={knn_k}"
        print(f"\n[Eval] {label} ...")
        metrics = run_single_eval(
            encoder_fn, method, pca, knn_k,
            fit_images, in_dist_eval, ood_eval,
            verbose=False,
        )
        results.append(metrics)

    # Print results table
    print("\n" + "=" * 70)
    print(f"  {'Config':<30s} {'AUROC':>7s} {'FPR@95':>8s} "
          f"{'FPR_def':>8s} {'p99_ms':>8s} {'µ_in':>8s} {'µ_ood':>8s}")
    print("  " + "-" * 68)
    for r in results:
        label = f"{r['method']} pca={r['pca_components']} k={r['knn_k']}"
        print(f"  {label:<30s} {r['auroc']:>7.4f} {r['fpr_at_tpr95']:>8.4f} "
              f"{r['fpr_default_threshold']:>8.4f} {r['latency_p99_ms']:>8.3f} "
              f"{r['mean_in_score']:>8.2f} {r['mean_ood_score']:>8.2f}")
    print("=" * 70)

    # Save to JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Eval] results saved → {args.output}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OOD Detection Evaluation")
    p.add_argument("--mode", choices=["synthetic", "lerobot", "robomimic", "disk"],
                   default="synthetic",
                   help="Frame source: synthetic, lerobot, robomimic, or disk")
    p.add_argument("--encoder", choices=["proxy", "dinov2"], default="proxy",
                   help="Encoder to use")
    p.add_argument("--n-fit", type=int, default=200,
                   help="Number of frames to fit detector")
    p.add_argument("--n-eval", type=int, default=100,
                   help="Number of eval frames per class")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweep", action="store_true",
                   help="Sweep over PCA and KNN-k configurations")
    p.add_argument("--output", type=str, default=None,
                   help="Save results to JSON file")
    p.add_argument("--dataset-repo", type=str, default="lerobot/pusht_image",
                   help="LeRobot dataset repo for in-dist (lerobot mode)")
    p.add_argument("--mild-ood-repo", type=str,
                   default="lerobot/utokyo_xarm_pick_and_place",
                   help="LeRobot dataset repo for mild OOD (lerobot mode)")
    p.add_argument("--dataset-path", type=str, default=None,
                   help="Path to HDF5 file (robomimic) or image dir (disk)")
    p.add_argument("--max-frames", type=int, default=1000,
                   help="Max frames to cache per OOD level")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
