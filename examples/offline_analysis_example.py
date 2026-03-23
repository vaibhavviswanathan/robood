"""
Offline Analysis Example — Failure Mode Identification
=======================================================
Demonstrates:
  1. Generating synthetic takeover episodes
  2. Logging them via TakeoverLogger
  3. Clustering with FailureModeIdentifier
  4. Heuristic interrogation (no API key needed)
  5. Generating and saving a collection brief

Usage:
    python examples/offline_analysis_example.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from vla_ood_detector import VLAOODDetector, OODResult
from vla_ood_demo import SyntheticFrameSource, make_proxy_encoder
from vla_failure_mode_identifier import TakeoverLogger, FailureModeIdentifier


def main():
    rng = np.random.default_rng(42)
    source = SyntheticFrameSource(rng)
    encoder_fn = make_proxy_encoder()

    print("\n" + "=" * 60)
    print("  Offline Analysis — Failure Mode Identification")
    print("=" * 60)

    # 1. Fit detector
    print("\n[1] Fitting detector ...")
    fit_images = [source.in_dist_frame() for _ in range(200)]
    detector = VLAOODDetector(
        encoder_fn=encoder_fn,
        method="mahalanobis",
        threshold_percentile=95.0,
        pca_components=32,
    )
    detector.fit(fit_images)

    # 2. Generate synthetic takeover episodes
    with tempfile.TemporaryDirectory() as log_dir:
        logger = TakeoverLogger(log_dir=log_dir)

        print("[2] Generating 40 synthetic takeover episodes ...")
        for i in range(40):
            if i < 20:
                frame = source.mild_ood_frame()
            else:
                frame = source.hard_ood_frame()

            result = detector.score(frame)
            logger.log_takeover(frame, result, metadata={"episode": i})

        # 3. Load episodes
        episodes = logger.load_all()
        print(f"    Loaded {len(episodes)} episodes")

        # 4. Cluster and interrogate
        print("\n[3] Clustering episodes ...")
        ref_frame = source.in_dist_frame()
        identifier = FailureModeIdentifier(
            in_dist_reference=ref_frame,
            min_cluster_size=3,
            api_key=None,  # heuristic fallback
        )
        clusters = identifier.cluster(episodes)
        print(f"    Found {len(clusters)} clusters")

        print("\n[4] Interrogating clusters (heuristic mode) ...")
        identifier.interrogate()

        # 5. Print and save brief
        identifier.print_brief()

        brief_path = Path(log_dir) / "failure_modes.json"
        identifier.save_brief(str(brief_path))
        print(f"\n[5] Brief saved → {brief_path}")

        # 6. Plot embedding space
        plot_path = Path(log_dir) / "failure_embedding_space.png"
        identifier.plot_embedding_space(episodes, save_path=str(plot_path))

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
