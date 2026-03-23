"""
Guarded VLA Example — End-to-End Integration
=============================================
Demonstrates:
  1. Fitting an OOD detector on in-distribution frames
  2. Wrapping a mock VLA policy with OODGuardedVLA
  3. Running a simulated control loop with OOD injection
  4. Logging takeover events on OOD detection

Usage:
    python examples/guarded_vla_example.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure imports work from examples/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from vla_ood_detector import VLAOODDetector, OODGuardedVLA
from vla_ood_demo import SyntheticFrameSource, make_proxy_encoder
from vla_failure_mode_identifier import TakeoverLogger


# ---------------------------------------------------------------------------
# Mock VLA policy
# ---------------------------------------------------------------------------

class MockVLAPolicy:
    """Simulates a VLA policy that outputs 7-DoF actions."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        # Dummy action: small random delta-pose + gripper
        return self.rng.uniform(-0.01, 0.01, 7).astype(np.float32)


# ---------------------------------------------------------------------------
# Simulated control loop
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)
    source = SyntheticFrameSource(rng)
    encoder_fn = make_proxy_encoder()

    print("\n" + "=" * 60)
    print("  Guarded VLA — End-to-End Example")
    print("=" * 60)

    # 1. Fit detector
    print("\n[1] Fitting OOD detector on 200 in-dist frames ...")
    fit_images = [source.in_dist_frame() for _ in range(200)]
    detector = VLAOODDetector(
        encoder_fn=encoder_fn,
        method="mahalanobis",
        threshold_percentile=95.0,
        pca_components=32,
    )
    detector.fit(fit_images)

    # 2. Wrap mock policy
    policy = MockVLAPolicy()
    guarded = OODGuardedVLA(policy, detector, on_ood="halt")
    print("[2] OODGuardedVLA ready (on_ood='halt')")

    # 3. Set up takeover logger
    with tempfile.TemporaryDirectory() as log_dir:
        logger = TakeoverLogger(log_dir=log_dir)
        print(f"[3] TakeoverLogger → {log_dir}")

        # 4. Simulated control loop
        print("\n[4] Running 60-step control loop ...")
        print(f"    {'step':>4s}  {'phase':<15s}  {'score':>8s}  {'action'}")
        print("    " + "-" * 50)

        n_executed = 0
        n_halted = 0

        for step in range(60):
            # Phase schedule
            if step < 30:
                obs = source.in_dist_frame()
                phase = "in-dist"
            elif step < 45:
                obs = source.mild_ood_frame()
                phase = "mild OOD"
            else:
                obs = source.hard_ood_frame()
                phase = "hard OOD"

            action, result = guarded.step(obs)

            if action is not None:
                n_executed += 1
                action_str = f"exec  [{action[0]:+.4f}, ...]"
            else:
                n_halted += 1
                action_str = "HALT  (takeover logged)"
                logger.log_takeover(obs, result, metadata={"step": step, "phase": phase})

            print(f"    {step:4d}  {phase:<15s}  {result.score:8.2f}  {action_str}")

        # 5. Summary
        episodes = logger.load_all()
        print(f"\n[5] Summary:")
        print(f"    Steps executed: {n_executed}")
        print(f"    Steps halted:   {n_halted}")
        print(f"    Takeover logs:  {len(episodes)}")
        print("=" * 60)


if __name__ == "__main__":
    main()
