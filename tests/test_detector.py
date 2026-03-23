"""Tests for vla_ood_detector module."""

import tempfile
import numpy as np
import pytest

from vla_ood_detector import VLAOODDetector, OODResult, OODGuardedVLA


# --- helpers ---

def make_dummy_encoder(dim=32, seed=0):
    """Random projection encoder for testing."""
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((224 * 224 * 3, dim)).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)

    def encode(image):
        flat = np.asarray(image, dtype=np.float32).ravel()
        if len(flat) != proj.shape[0]:
            flat = np.resize(flat, proj.shape[0])
        return flat @ proj

    return encode


def make_in_dist_images(n=50, seed=42):
    rng = np.random.default_rng(seed)
    return [rng.uniform(0.4, 0.6, (224, 224, 3)).astype(np.float32) for _ in range(n)]


def make_ood_images(n=20, seed=99):
    rng = np.random.default_rng(seed)
    return [rng.uniform(0.0, 1.0, (224, 224, 3)).astype(np.float32) for _ in range(n)]


# --- OODResult ---

class TestOODResult:
    def test_fields(self):
        r = OODResult(score=1.5, is_ood=True, method="mahalanobis",
                      threshold=1.0, embedding=np.zeros(32))
        assert r.score == 1.5
        assert r.is_ood is True
        assert r.method == "mahalanobis"
        assert r.threshold == 1.0
        assert r.embedding.shape == (32,)


# --- VLAOODDetector (Mahalanobis) ---

class TestMahalanobis:
    @pytest.fixture
    def detector(self):
        enc = make_dummy_encoder()
        det = VLAOODDetector(enc, method="mahalanobis", threshold_percentile=95.0,
                             pca_components=None)
        det.fit(make_in_dist_images(), verbose=False)
        return det

    def test_fit_sets_threshold(self, detector):
        assert detector._is_fitted
        assert detector._threshold is not None
        assert detector._threshold > 0

    def test_score_returns_ood_result(self, detector):
        img = make_in_dist_images(1)[0]
        r = detector.score(img)
        assert isinstance(r, OODResult)
        assert r.method == "mahalanobis"
        assert isinstance(r.score, float)
        assert isinstance(r.is_ood, bool)

    def test_ood_scores_higher(self, detector):
        in_scores = [detector.score(img).score for img in make_in_dist_images(20, seed=77)]
        ood_scores = [detector.score(img).score for img in make_ood_images(20)]
        assert np.mean(ood_scores) > np.mean(in_scores)

    def test_save_load_roundtrip(self, detector):
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            detector.save(f.name)
            det2 = VLAOODDetector(make_dummy_encoder(), method="mahalanobis")
            det2.load(f.name)

            img = make_in_dist_images(1)[0]
            r1 = detector.score(img)
            r2 = det2.score(img)
            assert abs(r1.score - r2.score) < 1e-4
            assert r1.is_ood == r2.is_ood


# --- VLAOODDetector (KNN) ---

class TestKNN:
    @pytest.fixture
    def detector(self):
        enc = make_dummy_encoder()
        det = VLAOODDetector(enc, method="knn", knn_k=5,
                             threshold_percentile=95.0, pca_components=None)
        det.fit(make_in_dist_images(), verbose=False)
        return det

    def test_fit_sets_threshold(self, detector):
        assert detector._is_fitted
        assert detector._threshold > 0

    def test_score_returns_ood_result(self, detector):
        r = detector.score(make_in_dist_images(1)[0])
        assert r.method == "knn"

    def test_ood_scores_higher(self, detector):
        in_scores = [detector.score(img).score for img in make_in_dist_images(20, seed=77)]
        ood_scores = [detector.score(img).score for img in make_ood_images(20)]
        assert np.mean(ood_scores) > np.mean(in_scores)

    def test_save_load_roundtrip(self, detector):
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            detector.save(f.name)
            det2 = VLAOODDetector(make_dummy_encoder(), method="knn")
            det2.load(f.name)

            img = make_in_dist_images(1)[0]
            r1 = detector.score(img)
            r2 = det2.score(img)
            assert abs(r1.score - r2.score) < 1e-4


# --- PCA ---

class TestPCA:
    def test_pca_reduces_dimensions(self):
        enc = make_dummy_encoder(dim=64)
        det = VLAOODDetector(enc, method="mahalanobis", pca_components=8)
        det.fit(make_in_dist_images(), verbose=False)
        r = det.score(make_in_dist_images(1)[0])
        assert r.embedding.shape == (8,)

    def test_pca_save_load(self):
        enc = make_dummy_encoder(dim=64)
        det = VLAOODDetector(enc, method="mahalanobis", pca_components=8)
        det.fit(make_in_dist_images(), verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            det.save(f.name)
            det2 = VLAOODDetector(make_dummy_encoder(dim=64), method="mahalanobis")
            det2.load(f.name)

            img = make_in_dist_images(1)[0]
            r1 = det.score(img)
            r2 = det2.score(img)
            assert abs(r1.score - r2.score) < 1e-4


# --- tune_threshold ---

class TestTuneThreshold:
    def test_tune_auroc(self):
        enc = make_dummy_encoder()
        det = VLAOODDetector(enc, method="mahalanobis")
        det.fit(make_in_dist_images(), verbose=False)
        old_t = det._threshold
        new_t = det.tune_threshold(make_in_dist_images(30, seed=11),
                                   make_ood_images(30), metric="auroc")
        assert isinstance(new_t, float)
        assert new_t > 0

    def test_tune_f1(self):
        enc = make_dummy_encoder()
        det = VLAOODDetector(enc, method="mahalanobis")
        det.fit(make_in_dist_images(), verbose=False)
        new_t = det.tune_threshold(make_in_dist_images(30, seed=11),
                                   make_ood_images(30), metric="f1")
        assert isinstance(new_t, float)
        assert new_t > 0


# --- OODGuardedVLA ---

class TestOODGuardedVLA:
    def test_halt_mode(self):
        enc = make_dummy_encoder()
        det = VLAOODDetector(enc, method="mahalanobis", threshold_percentile=50.0)
        det.fit(make_in_dist_images(), verbose=False)

        class MockPolicy:
            def predict(self, obs):
                return np.zeros(7)

        guarded = OODGuardedVLA(MockPolicy(), det, on_ood="halt")
        # Score an OOD image — should halt
        action, result = guarded.step(make_ood_images(1)[0])
        if result.is_ood:
            assert action is None

    def test_log_only_mode(self):
        enc = make_dummy_encoder()
        det = VLAOODDetector(enc, method="mahalanobis", threshold_percentile=50.0)
        det.fit(make_in_dist_images(), verbose=False)

        class MockPolicy:
            def predict(self, obs):
                return np.zeros(7)

        guarded = OODGuardedVLA(MockPolicy(), det, on_ood="log_only")
        action, result = guarded.step(make_ood_images(1)[0])
        # log_only always returns an action
        assert action is not None

    def test_invalid_on_ood(self):
        enc = make_dummy_encoder()
        det = VLAOODDetector(enc, method="mahalanobis")
        class MockPolicy:
            def predict(self, obs):
                return np.zeros(7)
        with pytest.raises(ValueError):
            OODGuardedVLA(MockPolicy(), det, on_ood="invalid")


# --- edge cases ---

class TestEdgeCases:
    def test_invalid_method(self):
        with pytest.raises(ValueError):
            VLAOODDetector(make_dummy_encoder(), method="invalid")

    def test_score_before_fit(self):
        det = VLAOODDetector(make_dummy_encoder(), method="mahalanobis")
        with pytest.raises(RuntimeError):
            det.score(make_in_dist_images(1)[0])

    def test_save_before_fit(self):
        det = VLAOODDetector(make_dummy_encoder(), method="mahalanobis")
        with pytest.raises(RuntimeError):
            det.save("/tmp/test.npz")
