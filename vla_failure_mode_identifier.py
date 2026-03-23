"""
VLA Failure Mode Identifier
============================
Offline analysis pipeline: logs takeover episodes, clusters them into
failure modes, interrogates via VLM, and produces collection briefs.

Classes:
    TakeoverEpisode     — single takeover event
    TakeoverLogger      — persists episodes to disk
    FailureCluster      — cluster of related failure episodes
    FailureModeIdentifier — clustering + VLM interrogation + brief generation
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class TakeoverEpisode:
    """A single takeover event with frame, embedding, and metadata."""
    frame: np.ndarray
    embedding: np.ndarray
    ood_score: float
    timestamp: float
    metadata: dict = field(default_factory=dict)


@dataclass
class FailureCluster:
    """A cluster of related failure episodes."""
    cluster_id: int
    episodes: list
    representative_frame: np.ndarray
    centroid: np.ndarray
    failure_mode: str = ""
    description: str = ""
    collection_brief: str = ""
    suggested_n_demos: int = 50

    def to_dict(self) -> dict:
        return {
            "cluster_id": int(self.cluster_id),
            "n_episodes": len(self.episodes),
            "failure_mode": self.failure_mode,
            "description": self.description,
            "collection_brief": self.collection_brief,
            "suggested_n_demos": int(self.suggested_n_demos),
        }


class TakeoverLogger:
    """
    Persists takeover episodes to disk as compressed .npz files.

    Parameters
    ----------
    log_dir : str
        Directory to store episode files.
    """

    def __init__(self, log_dir: str = "./takeover_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_takeover(self, frame: np.ndarray, ood_result, metadata: dict = None) -> str:
        """
        Log a takeover episode to disk.

        Parameters
        ----------
        frame : np.ndarray
            Camera frame at moment of takeover
        ood_result : OODResult
            Detection result from the OOD detector
        metadata : dict, optional
            Additional metadata (task name, robot state, etc.)

        Returns
        -------
        str : path to saved file
        """
        ts = time.time()
        filename = f"takeover_{ts:.6f}.npz"
        path = self.log_dir / filename

        save_data = {
            "frame": np.asarray(frame, dtype=np.float32),
            "embedding": np.asarray(ood_result.embedding, dtype=np.float32),
            "ood_score": np.array(ood_result.score, dtype=np.float64),
            "timestamp": np.array(ts, dtype=np.float64),
        }
        if metadata:
            save_data["metadata_json"] = np.array(json.dumps(metadata))

        np.savez_compressed(str(path), **save_data)
        return str(path)

    def load_all(self) -> list[TakeoverEpisode]:
        """Load all takeover episodes from the log directory."""
        episodes = []
        for f in sorted(self.log_dir.glob("takeover_*.npz")):
            d = np.load(str(f), allow_pickle=False)
            meta = {}
            if "metadata_json" in d:
                meta = json.loads(str(d["metadata_json"]))
            episodes.append(TakeoverEpisode(
                frame=d["frame"],
                embedding=d["embedding"],
                ood_score=float(d["ood_score"]),
                timestamp=float(d["timestamp"]),
                metadata=meta,
            ))
        return episodes


class FailureModeIdentifier:
    """
    Clusters takeover episodes into failure modes and generates
    actionable collection briefs via VLM interrogation.

    Parameters
    ----------
    in_dist_reference : np.ndarray
        A representative in-distribution frame for contrastive prompting.
    min_cluster_size : int
        Minimum cluster size for HDBSCAN or k-means.
    api_key : str or None
        Anthropic API key for VLM interrogation. None uses heuristic fallback.
    """

    def __init__(
        self,
        in_dist_reference: np.ndarray,
        min_cluster_size: int = 5,
        api_key: Optional[str] = None,
    ):
        self.in_dist_reference = in_dist_reference
        self.min_cluster_size = min_cluster_size
        self.api_key = api_key
        self.clusters: list[FailureCluster] = []

    def cluster(self, episodes: list[TakeoverEpisode]) -> list[FailureCluster]:
        """
        Cluster episodes by embedding similarity.
        Uses HDBSCAN if available, falls back to k-means with silhouette selection.
        """
        if len(episodes) < 2:
            if episodes:
                self.clusters = [FailureCluster(
                    cluster_id=0,
                    episodes=episodes,
                    representative_frame=episodes[0].frame,
                    centroid=episodes[0].embedding,
                )]
            return self.clusters

        embeddings = np.stack([e.embedding for e in episodes])

        labels = self._cluster_embeddings(embeddings)

        unique_labels = sorted(set(labels))
        # Remove noise label (-1) for building clusters
        if -1 in unique_labels:
            unique_labels.remove(-1)

        self.clusters = []
        for cid in unique_labels:
            mask = labels == cid
            cluster_episodes = [e for e, m in zip(episodes, mask) if m]
            cluster_embeds = embeddings[mask]
            centroid = cluster_embeds.mean(axis=0)

            # Representative frame = closest to centroid
            dists = np.linalg.norm(cluster_embeds - centroid, axis=1)
            rep_idx = int(np.argmin(dists))

            self.clusters.append(FailureCluster(
                cluster_id=cid,
                episodes=cluster_episodes,
                representative_frame=cluster_episodes[rep_idx].frame,
                centroid=centroid,
            ))

        # Sort by cluster size descending (most common failure first)
        self.clusters.sort(key=lambda c: len(c.episodes), reverse=True)
        return self.clusters

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Try HDBSCAN, fallback to k-means with silhouette-based k."""
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(embeddings)
            if len(set(labels) - {-1}) >= 1:
                return labels
        except ImportError:
            pass

        # Fallback: k-means with silhouette-based k selection
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        max_k = min(10, len(embeddings) // max(self.min_cluster_size, 2))
        max_k = max(max_k, 2)

        best_k, best_score = 2, -1.0
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labs = km.fit_predict(embeddings)
            if len(set(labs)) < 2:
                continue
            score = silhouette_score(embeddings, labs)
            if score > best_score:
                best_score = score
                best_k = k

        km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        return km.fit_predict(embeddings)

    def interrogate(self, use_contrastive: bool = True) -> list[FailureCluster]:
        """
        Generate failure mode descriptions for each cluster.
        Uses Claude API if api_key is set, otherwise heuristic fallback.
        """
        for cluster in self.clusters:
            if self.api_key:
                self._interrogate_vlm(cluster, use_contrastive)
            else:
                self._interrogate_heuristic(cluster)
        return self.clusters

    def _interrogate_vlm(self, cluster: FailureCluster, use_contrastive: bool):
        """Call Claude API with contrastive or caption-only prompt."""
        try:
            import anthropic
            import base64
            from io import BytesIO

            client = anthropic.Anthropic(api_key=self.api_key)

            # Encode frames as base64 PNG
            def frame_to_base64(frame):
                from PIL import Image
                img = Image.fromarray(
                    (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                    if frame.max() <= 1.0
                    else frame.astype(np.uint8)
                )
                buf = BytesIO()
                img.save(buf, format="PNG")
                return base64.standard_b64encode(buf.getvalue()).decode()

            content = []
            if use_contrastive:
                content.append({
                    "type": "text",
                    "text": "Here is a NORMAL in-distribution frame from a robot workspace:"
                })
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": frame_to_base64(self.in_dist_reference),
                    }
                })
                content.append({
                    "type": "text",
                    "text": "Here is an OUT-OF-DISTRIBUTION frame that caused a failure:"
                })
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": frame_to_base64(cluster.representative_frame),
                    }
                })
                content.append({
                    "type": "text",
                    "text": (
                        f"This cluster has {len(cluster.episodes)} similar failure episodes.\n\n"
                        "Compare the two frames. What specifically changed in the OOD frame "
                        "that would cause the robot's vision-language-action policy to fail?\n\n"
                        "Respond ONLY with valid JSON:\n"
                        '{"failure_mode": "short name", '
                        '"description": "detailed explanation of why this causes failure", '
                        '"collection_brief": "what data to collect to fix this", '
                        '"suggested_n_demos": <integer>}'
                    ),
                })
            else:
                content.append({
                    "type": "text",
                    "text": "Here is an OUT-OF-DISTRIBUTION frame from a robot workspace that caused a failure:"
                })
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": frame_to_base64(cluster.representative_frame),
                    }
                })
                content.append({
                    "type": "text",
                    "text": (
                        f"This cluster has {len(cluster.episodes)} similar failure episodes.\n\n"
                        "What about this frame might cause a robot's vision policy to fail?\n\n"
                        "Respond ONLY with valid JSON:\n"
                        '{"failure_mode": "short name", '
                        '"description": "detailed explanation", '
                        '"collection_brief": "what data to collect", '
                        '"suggested_n_demos": <integer>}'
                    ),
                })

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                messages=[{"role": "user", "content": content}],
            )

            text = response.content[0].text.strip()
            # Extract JSON from response
            if "{" in text:
                json_str = text[text.index("{"):text.rindex("}") + 1]
                data = json.loads(json_str)
                cluster.failure_mode = data.get("failure_mode", "unknown")
                cluster.description = data.get("description", "")
                cluster.collection_brief = data.get("collection_brief", "")
                cluster.suggested_n_demos = data.get("suggested_n_demos", 50)
        except Exception as e:
            print(f"[FailureModeIdentifier] VLM call failed: {e}")
            self._interrogate_heuristic(cluster)

    def _interrogate_heuristic(self, cluster: FailureCluster):
        """Generate heuristic failure description without VLM."""
        n = len(cluster.episodes)
        avg_score = np.mean([e.ood_score for e in cluster.episodes])

        # Compute simple visual statistics for differentiation
        frames = np.stack([e.frame for e in cluster.episodes])
        mean_brightness = float(frames.mean())
        mean_contrast = float(frames.std())

        ref_brightness = float(self.in_dist_reference.mean())
        ref_contrast = float(self.in_dist_reference.std())

        diffs = []
        if abs(mean_brightness - ref_brightness) > 0.1:
            direction = "brighter" if mean_brightness > ref_brightness else "darker"
            diffs.append(f"significantly {direction} than normal")
        if abs(mean_contrast - ref_contrast) > 0.05:
            direction = "higher" if mean_contrast > ref_contrast else "lower"
            diffs.append(f"{direction} contrast")

        if not diffs:
            diffs.append("visual distribution shift")

        cluster.failure_mode = f"cluster_{cluster.cluster_id}: {', '.join(diffs)}"
        cluster.description = (
            f"Cluster of {n} episodes with mean OOD score {avg_score:.2f}. "
            f"Mean brightness={mean_brightness:.3f} (ref={ref_brightness:.3f}), "
            f"contrast={mean_contrast:.3f} (ref={ref_contrast:.3f})."
        )
        cluster.collection_brief = (
            f"Collect {max(50, n * 3)} demonstrations covering conditions: {', '.join(diffs)}."
        )
        cluster.suggested_n_demos = max(50, n * 3)

    def print_brief(self):
        """Print ranked failure mode brief to console."""
        print("\n" + "=" * 60)
        print("  FAILURE MODE BRIEF")
        print("=" * 60)
        for i, c in enumerate(self.clusters):
            print(f"\n  #{i+1}  {c.failure_mode}")
            print(f"      Episodes: {len(c.episodes)}")
            print(f"      {c.description}")
            print(f"      Action: {c.collection_brief}")
            print(f"      Suggested demos: {c.suggested_n_demos}")
        print("\n" + "=" * 60)

    def save_brief(self, path: str):
        """Export failure mode brief as JSON."""
        brief = {
            "n_clusters": len(self.clusters),
            "total_episodes": sum(len(c.episodes) for c in self.clusters),
            "clusters": [c.to_dict() for c in self.clusters],
        }
        with open(path, "w") as f:
            json.dump(brief, f, indent=2)

    def plot_embedding_space(
        self,
        episodes: list[TakeoverEpisode],
        save_path: str = "failure_embedding_space.png",
    ):
        """t-SNE scatter plot of episode embeddings colored by cluster."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
        except ImportError:
            print("[FailureModeIdentifier] matplotlib or sklearn not available for plotting")
            return

        embeddings = np.stack([e.embedding for e in episodes])

        perplexity = min(30, max(2, len(episodes) - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords = tsne.fit_transform(embeddings)

        # Build label array from clusters
        episode_to_cluster = {}
        for c in self.clusters:
            for ep in c.episodes:
                episode_to_cluster[id(ep)] = c.cluster_id

        labels = [episode_to_cluster.get(id(e), -1) for e in episodes]

        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

        for i, lab in enumerate(unique_labels):
            mask = np.array(labels) == lab
            name = f"Cluster {lab}" if lab >= 0 else "Noise"
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[colors[i]], label=name, s=30, alpha=0.7)

        ax.set_title("Failure Episode Embedding Space (t-SNE)")
        ax.legend(fontsize=8)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[FailureModeIdentifier] saved plot → {save_path}")
