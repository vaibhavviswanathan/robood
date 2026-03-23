"""
VLA OOD Detector — Core Module
===============================
Out-of-distribution detection for Vision-Language-Action robot policies.

Classes:
    OODResult           — detection result dataclass
    VLAOODDetector      — fits density model on in-dist embeddings, scores new frames
    OODGuardedVLA       — wraps any VLA policy with a pre-action OOD gate
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class OODResult:
    """Result of scoring a single frame for OOD-ness."""
    score: float
    is_ood: bool
    method: str
    threshold: float
    embedding: np.ndarray = field(repr=False)


class VLAOODDetector:
    """
    Core OOD detector — fits a density model on in-distribution embeddings
    and scores new frames via Mahalanobis distance or KNN distance.

    Parameters
    ----------
    encoder_fn : callable
        image (H×W×3 float32 0-1) → 1-D float32 embedding
    method : str
        "mahalanobis" or "knn"
    knn_k : int
        Number of neighbors for KNN scoring
    threshold_percentile : float
        Percentile of in-dist scores used as default threshold τ
    pca_components : int or None
        If set, apply PCA dimensionality reduction before scoring
    """

    def __init__(
        self,
        encoder_fn: Callable,
        method: str = "mahalanobis",
        knn_k: int = 5,
        threshold_percentile: float = 95.0,
        pca_components: Optional[int] = None,
    ):
        if method not in ("mahalanobis", "knn"):
            raise ValueError(f"method must be 'mahalanobis' or 'knn', got '{method}'")
        self.encoder_fn = encoder_fn
        self.method = method
        self.knn_k = knn_k
        self.threshold_percentile = threshold_percentile
        self.pca_components = pca_components

        self._is_fitted = False
        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._threshold: Optional[float] = None
        self._pca = None
        self._train_embeddings: Optional[np.ndarray] = None

    def fit(self, images: list, verbose: bool = True) -> "VLAOODDetector":
        """Fit detector on a list of in-distribution images."""
        if verbose:
            print(f"[Detector] encoding {len(images)} images ...")
        embeddings = np.stack([self._encode(img) for img in images])

        if self.pca_components is not None:
            embeddings = self._fit_pca(embeddings, verbose=verbose)

        if self.method == "mahalanobis":
            self._mean = embeddings.mean(axis=0)
            cov = np.cov(embeddings, rowvar=False)
            cov += np.eye(cov.shape[0]) * 1e-5
            self._cov_inv = np.linalg.inv(cov)
        else:  # knn
            self._train_embeddings = embeddings.copy()

        scores = self._score_embeddings(embeddings)
        self._threshold = float(np.percentile(scores, self.threshold_percentile))
        self._is_fitted = True

        if verbose:
            print(f"[Detector] fitted  method={self.method}  τ={self._threshold:.3f}")
        return self

    def score(self, image: np.ndarray) -> OODResult:
        """Score a single image for OOD-ness."""
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        z = self._encode(image)
        if self._pca is not None:
            z = self._pca.transform(z[None])[0]
        s = self._score_single(z)
        return OODResult(
            score=float(s),
            is_ood=s >= self._threshold,
            method=self.method,
            threshold=self._threshold,
            embedding=z,
        )

    def save(self, path: str) -> None:
        """Persist detector state to a .npz file."""
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        data = {
            "method": np.array(self.method),
            "knn_k": np.array(self.knn_k),
            "threshold_percentile": np.array(self.threshold_percentile),
            "threshold": np.array(self._threshold),
        }
        if self._mean is not None:
            data["mean"] = self._mean
        if self._cov_inv is not None:
            data["cov_inv"] = self._cov_inv
        if self._train_embeddings is not None:
            data["train_embeddings"] = self._train_embeddings
        if self._pca is not None:
            data["pca_components"] = np.array(self.pca_components)
            data["pca_mean"] = self._pca.mean_
            data["pca_components_matrix"] = self._pca.components_
            data["pca_explained_variance"] = self._pca.explained_variance_
            data["pca_whiten"] = np.array(self._pca.whiten)
        np.savez_compressed(path, **data)

    def load(self, path: str) -> "VLAOODDetector":
        """Load detector state from a .npz file."""
        d = np.load(path, allow_pickle=False)
        self.method = str(d["method"])
        self.knn_k = int(d["knn_k"])
        self.threshold_percentile = float(d["threshold_percentile"])
        self._threshold = float(d["threshold"])

        if "mean" in d:
            self._mean = d["mean"]
        if "cov_inv" in d:
            self._cov_inv = d["cov_inv"]
        if "train_embeddings" in d:
            self._train_embeddings = d["train_embeddings"]
        if "pca_components" in d:
            from sklearn.decomposition import PCA
            n_comp = int(d["pca_components"])
            whiten = bool(d["pca_whiten"]) if "pca_whiten" in d else False
            self.pca_components = n_comp
            self._pca = PCA(n_components=n_comp, whiten=whiten)
            self._pca.mean_ = d["pca_mean"]
            self._pca.components_ = d["pca_components_matrix"]
            self._pca.explained_variance_ = d["pca_explained_variance"]
            self._pca.n_components_ = n_comp
            self._pca.n_features_in_ = len(d["pca_mean"])
            self._pca.whiten = whiten

        self._is_fitted = True
        return self

    def tune_threshold(
        self,
        in_dist_images: list,
        ood_images: list,
        metric: str = "auroc",
    ) -> float:
        """
        Tune threshold using labelled in-dist and OOD images.

        Parameters
        ----------
        metric : str
            "auroc" — Youden's J optimal point
            "f1" — threshold maximizing F1 score

        Returns the new threshold τ.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        in_scores = np.array([self.score(img).score for img in in_dist_images])
        ood_scores = np.array([self.score(img).score for img in ood_images])

        scores = np.concatenate([in_scores, ood_scores])
        labels = np.concatenate([
            np.zeros(len(in_scores)),
            np.ones(len(ood_scores)),
        ])

        from sklearn.metrics import roc_curve, f1_score

        if metric == "auroc":
            fpr, tpr, thresholds = roc_curve(labels, scores)
            # Youden's J statistic
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            self._threshold = float(thresholds[best_idx])
        elif metric == "f1":
            # Search over candidate thresholds
            candidates = np.percentile(scores, np.linspace(1, 99, 200))
            best_f1, best_t = -1.0, float(np.median(scores))
            for t in candidates:
                preds = (scores >= t).astype(int)
                f = f1_score(labels, preds)
                if f > best_f1:
                    best_f1 = f
                    best_t = float(t)
            self._threshold = best_t
        else:
            raise ValueError(f"metric must be 'auroc' or 'f1', got '{metric}'")

        return self._threshold

    # --- internal helpers ---

    def _encode(self, image: np.ndarray) -> np.ndarray:
        z = self.encoder_fn(image)
        if hasattr(z, "numpy"):
            z = z.numpy()
        z = np.asarray(z, dtype=np.float32).ravel()
        norm = np.linalg.norm(z)
        return z / norm if norm > 1e-8 else z

    def _fit_pca(self, embeddings: np.ndarray, verbose: bool = True) -> np.ndarray:
        from sklearn.decomposition import PCA
        n = min(self.pca_components, embeddings.shape[1], embeddings.shape[0])
        self._pca = PCA(n_components=n, whiten=True)
        result = self._pca.fit_transform(embeddings)
        if verbose:
            var = self._pca.explained_variance_ratio_.sum()
            print(f"[Detector] PCA {embeddings.shape[1]} → {n}  "
                  f"(explained variance: {var:.1%})")
        return result

    def _score_single(self, z: np.ndarray) -> float:
        if self.method == "mahalanobis":
            return self._mahalanobis(z)
        else:
            return self._knn_score(z)

    def _score_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        return np.array([self._score_single(z) for z in embeddings])

    def _mahalanobis(self, z: np.ndarray) -> float:
        diff = z - self._mean
        return float(diff @ self._cov_inv @ diff)

    def _knn_score(self, z: np.ndarray) -> float:
        dists = np.linalg.norm(self._train_embeddings - z, axis=1)
        k = min(self.knn_k, len(dists))
        topk = np.partition(dists, k)[:k]
        return float(np.mean(topk))


class OODGuardedVLA:
    """
    Wraps a VLA policy with a pre-action OOD gate.

    Parameters
    ----------
    vla_policy : object
        Must implement .predict(obs) → action
    ood_detector : VLAOODDetector
        Fitted OOD detector
    on_ood : str
        "halt" — return None action when OOD detected
        "log_only" — always execute policy, just tag the result
    """

    def __init__(
        self,
        vla_policy,
        ood_detector: VLAOODDetector,
        on_ood: str = "halt",
    ):
        if on_ood not in ("halt", "log_only"):
            raise ValueError(f"on_ood must be 'halt' or 'log_only', got '{on_ood}'")
        self.vla_policy = vla_policy
        self.ood_detector = ood_detector
        self.on_ood = on_ood

    def step(self, obs: np.ndarray):
        """
        Score the observation and conditionally execute the policy.

        Returns
        -------
        (action or None, OODResult)
        """
        result = self.ood_detector.score(obs)
        if result.is_ood and self.on_ood == "halt":
            return None, result
        action = self.vla_policy.predict(obs)
        return action, result
