"""Tests for evaluate_ood module."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_ood import compute_metrics, run_single_eval
from vla_ood_detector import VLAOODDetector


def make_dummy_encoder(dim=32, seed=0):
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((224 * 224 * 3, dim)).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)

    def encode(image):
        flat = np.asarray(image, dtype=np.float32).ravel()
        if len(flat) != proj.shape[0]:
            flat = np.resize(flat, proj.shape[0])
        return flat @ proj

    return encode


def make_images(n, brightness, seed):
    rng = np.random.default_rng(seed)
    return [rng.uniform(brightness - 0.1, brightness + 0.1,
                        (224, 224, 3)).astype(np.float32) for _ in range(n)]


class TestComputeMetrics:
    def test_returns_expected_keys(self):
        enc = make_dummy_encoder()
        det = VLAOODDetector(enc, method="mahalanobis")
        det.fit(make_images(50, 0.5, 42), verbose=False)

        metrics = compute_metrics(det, make_images(20, 0.5, 11), make_images(20, 0.1, 99))
        expected_keys = {"auroc", "fpr_at_tpr95", "fpr_default_threshold",
                         "latency_p99_ms", "latency_mean_ms", "threshold",
                         "n_in_dist", "n_ood", "mean_in_score", "mean_ood_score"}
        assert expected_keys <= set(metrics.keys())

    def test_auroc_reasonable(self):
        enc = make_dummy_encoder()
        det = VLAOODDetector(enc, method="mahalanobis")
        det.fit(make_images(50, 0.5, 42), verbose=False)

        metrics = compute_metrics(det, make_images(20, 0.5, 11), make_images(20, 0.1, 99))
        assert 0.0 <= metrics["auroc"] <= 1.0
        # With well-separated data, AUROC should be decent
        assert metrics["auroc"] > 0.5

    def test_latency_positive(self):
        enc = make_dummy_encoder()
        det = VLAOODDetector(enc, method="mahalanobis")
        det.fit(make_images(50, 0.5, 42), verbose=False)

        metrics = compute_metrics(det, make_images(10, 0.5, 11), make_images(10, 0.1, 99))
        assert metrics["latency_p99_ms"] > 0
        assert metrics["latency_mean_ms"] > 0


class TestRunSingleEval:
    def test_mahalanobis_eval(self):
        enc = make_dummy_encoder()
        r = run_single_eval(
            enc, "mahalanobis", None, 5,
            make_images(50, 0.5, 42),
            make_images(20, 0.5, 11),
            make_images(20, 0.1, 99),
            verbose=False,
        )
        assert r["method"] == "mahalanobis"
        assert 0.0 <= r["auroc"] <= 1.0

    def test_knn_eval(self):
        enc = make_dummy_encoder()
        r = run_single_eval(
            enc, "knn", None, 5,
            make_images(50, 0.5, 42),
            make_images(20, 0.5, 11),
            make_images(20, 0.1, 99),
            verbose=False,
        )
        assert r["method"] == "knn"
        assert r["knn_k"] == 5

    def test_with_pca(self):
        enc = make_dummy_encoder()
        r = run_single_eval(
            enc, "mahalanobis", 8, 5,
            make_images(50, 0.5, 42),
            make_images(20, 0.5, 11),
            make_images(20, 0.1, 99),
            verbose=False,
        )
        assert r["pca_components"] == 8
