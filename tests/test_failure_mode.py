"""Tests for vla_failure_mode_identifier module."""

import json
import tempfile
import numpy as np
import pytest

from vla_failure_mode_identifier import (
    TakeoverEpisode,
    TakeoverLogger,
    FailureCluster,
    FailureModeIdentifier,
)
from vla_ood_detector import OODResult


# --- helpers ---

def make_ood_result(score=5.0, embedding_dim=32, seed=0):
    rng = np.random.default_rng(seed)
    return OODResult(
        score=score,
        is_ood=True,
        method="mahalanobis",
        threshold=3.0,
        embedding=rng.standard_normal(embedding_dim).astype(np.float32),
    )


def make_frame(seed=0, brightness=0.5):
    rng = np.random.default_rng(seed)
    return rng.uniform(brightness - 0.1, brightness + 0.1,
                       (64, 64, 3)).astype(np.float32)


def make_episodes(n=20, n_clusters=3, embed_dim=32, seed=42):
    """Generate synthetic episodes with cluster structure."""
    rng = np.random.default_rng(seed)
    episodes = []
    for i in range(n):
        cluster_id = i % n_clusters
        # Each cluster has a distinct centroid
        centroid = np.zeros(embed_dim, dtype=np.float32)
        centroid[cluster_id * 3:(cluster_id + 1) * 3] = 5.0
        embedding = centroid + rng.standard_normal(embed_dim).astype(np.float32) * 0.3

        brightness = 0.3 + cluster_id * 0.2
        episodes.append(TakeoverEpisode(
            frame=make_frame(seed=i, brightness=brightness),
            embedding=embedding,
            ood_score=float(rng.uniform(3.0, 10.0)),
            timestamp=float(i),
        ))
    return episodes


# --- TakeoverLogger ---

class TestTakeoverLogger:
    def test_log_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TakeoverLogger(log_dir=tmpdir)

            frame = make_frame(seed=1)
            result = make_ood_result(score=7.5, seed=1)
            path = logger.log_takeover(frame, result, metadata={"task": "lift"})
            assert path.endswith(".npz")

            episodes = logger.load_all()
            assert len(episodes) == 1
            ep = episodes[0]
            np.testing.assert_array_almost_equal(ep.frame, frame, decimal=5)
            np.testing.assert_array_almost_equal(ep.embedding, result.embedding, decimal=5)
            assert abs(ep.ood_score - 7.5) < 1e-6
            assert ep.metadata == {"task": "lift"}

    def test_multiple_episodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TakeoverLogger(log_dir=tmpdir)
            for i in range(5):
                logger.log_takeover(make_frame(seed=i), make_ood_result(seed=i))

            episodes = logger.load_all()
            assert len(episodes) == 5

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TakeoverLogger(log_dir=tmpdir)
            assert logger.load_all() == []

    def test_no_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TakeoverLogger(log_dir=tmpdir)
            logger.log_takeover(make_frame(), make_ood_result())
            ep = logger.load_all()[0]
            assert ep.metadata == {}


# --- FailureCluster ---

class TestFailureCluster:
    def test_to_dict(self):
        c = FailureCluster(
            cluster_id=0,
            episodes=[1, 2, 3],
            representative_frame=np.zeros((64, 64, 3)),
            centroid=np.zeros(32),
            failure_mode="test mode",
            description="test desc",
            collection_brief="collect more",
            suggested_n_demos=75,
        )
        d = c.to_dict()
        assert d["cluster_id"] == 0
        assert d["n_episodes"] == 3
        assert d["failure_mode"] == "test mode"
        assert d["suggested_n_demos"] == 75


# --- FailureModeIdentifier ---

class TestFailureModeIdentifier:
    def test_cluster_basic(self):
        episodes = make_episodes(n=30, n_clusters=3)
        ref_frame = make_frame(seed=999, brightness=0.5)
        identifier = FailureModeIdentifier(ref_frame, min_cluster_size=3)
        clusters = identifier.cluster(episodes)
        assert len(clusters) >= 1
        total = sum(len(c.episodes) for c in clusters)
        assert total == 30

    def test_cluster_single_episode(self):
        episodes = make_episodes(n=1, n_clusters=1)
        ref_frame = make_frame(seed=999)
        identifier = FailureModeIdentifier(ref_frame)
        clusters = identifier.cluster(episodes)
        assert len(clusters) == 1
        assert len(clusters[0].episodes) == 1

    def test_cluster_empty(self):
        ref_frame = make_frame(seed=999)
        identifier = FailureModeIdentifier(ref_frame)
        clusters = identifier.cluster([])
        assert len(clusters) == 0

    def test_heuristic_interrogation(self):
        episodes = make_episodes(n=15, n_clusters=2)
        ref_frame = make_frame(seed=999, brightness=0.5)
        identifier = FailureModeIdentifier(ref_frame, min_cluster_size=3, api_key=None)
        identifier.cluster(episodes)
        identifier.interrogate()

        for c in identifier.clusters:
            assert c.failure_mode != ""
            assert c.description != ""
            assert c.collection_brief != ""
            assert c.suggested_n_demos > 0

    def test_print_brief(self, capsys):
        episodes = make_episodes(n=15, n_clusters=2)
        ref_frame = make_frame(seed=999)
        identifier = FailureModeIdentifier(ref_frame, min_cluster_size=3)
        identifier.cluster(episodes)
        identifier.interrogate()
        identifier.print_brief()

        captured = capsys.readouterr()
        assert "FAILURE MODE BRIEF" in captured.out

    def test_save_brief(self):
        episodes = make_episodes(n=15, n_clusters=2)
        ref_frame = make_frame(seed=999)
        identifier = FailureModeIdentifier(ref_frame, min_cluster_size=3)
        identifier.cluster(episodes)
        identifier.interrogate()

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            identifier.save_brief(path)
            with open(path) as f:
                data = json.load(f)
            assert "n_clusters" in data
            assert "clusters" in data
            assert data["total_episodes"] == 15
        finally:
            import os
            os.unlink(path)

    def test_sorted_by_size(self):
        episodes = make_episodes(n=30, n_clusters=3)
        ref_frame = make_frame(seed=999)
        identifier = FailureModeIdentifier(ref_frame, min_cluster_size=3)
        identifier.cluster(episodes)
        sizes = [len(c.episodes) for c in identifier.clusters]
        assert sizes == sorted(sizes, reverse=True)
