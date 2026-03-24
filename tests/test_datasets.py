"""
Tests for datasets.py frame sources.

Uses only local/synthetic data — no network calls.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from frame_sources import DiskFrameSource, _resize_frame


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_dummy_pngs(directory: Path, n: int = 5, size: int = 64):
    """Write n dummy PNG files into directory."""
    from PIL import Image
    for i in range(n):
        img = Image.fromarray(
            np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        )
        img.save(directory / f"frame_{i:03d}.png")


# ---------------------------------------------------------------------------
# _resize_frame
# ---------------------------------------------------------------------------

def test_resize_frame_noop():
    frame = np.random.rand(224, 224, 3).astype(np.float32)
    out = _resize_frame(frame, 224)
    assert out is frame  # should return same object when no resize needed


def test_resize_frame_changes_size():
    frame = np.random.rand(100, 150, 3).astype(np.float32)
    out = _resize_frame(frame, 64)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


# ---------------------------------------------------------------------------
# DiskFrameSource
# ---------------------------------------------------------------------------

class TestDiskFrameSource:
    def test_load_and_sample(self, tmp_path):
        _make_dummy_pngs(tmp_path, n=5, size=32)
        src = DiskFrameSource(
            in_dist_dir=str(tmp_path),
            img_size=64,
            rng=np.random.default_rng(0),
        )
        frame = src.in_dist_frame()
        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.float32
        assert 0.0 <= frame.min() and frame.max() <= 1.0

    def test_mild_ood_fallback_jitter(self, tmp_path):
        """When no mild_ood_dir, should produce color-jittered in-dist frames."""
        _make_dummy_pngs(tmp_path, n=3, size=32)
        src = DiskFrameSource(
            in_dist_dir=str(tmp_path),
            img_size=32,
            rng=np.random.default_rng(42),
        )
        frame = src.mild_ood_frame()
        assert frame.shape == (32, 32, 3)
        assert frame.dtype == np.float32

    def test_separate_mild_ood_dir(self, tmp_path):
        in_dir = tmp_path / "in_dist"
        ood_dir = tmp_path / "mild_ood"
        in_dir.mkdir()
        ood_dir.mkdir()
        _make_dummy_pngs(in_dir, n=3, size=32)
        _make_dummy_pngs(ood_dir, n=3, size=32)
        src = DiskFrameSource(
            in_dist_dir=str(in_dir),
            mild_ood_dir=str(ood_dir),
            img_size=32,
            rng=np.random.default_rng(0),
        )
        assert src.in_dist_frame().shape == (32, 32, 3)
        assert src.mild_ood_frame().shape == (32, 32, 3)

    def test_hard_ood_from_dir(self, tmp_path):
        in_dir = tmp_path / "in"
        hard_dir = tmp_path / "hard"
        in_dir.mkdir()
        hard_dir.mkdir()
        _make_dummy_pngs(in_dir, n=2, size=32)
        _make_dummy_pngs(hard_dir, n=2, size=32)
        src = DiskFrameSource(
            in_dist_dir=str(in_dir),
            hard_ood_dir=str(hard_dir),
            img_size=32,
            rng=np.random.default_rng(0),
        )
        frame = src.hard_ood_frame()
        assert frame.shape == (32, 32, 3)

    def test_empty_dir_raises(self, tmp_path):
        src = DiskFrameSource(
            in_dist_dir=str(tmp_path),
            rng=np.random.default_rng(0),
        )
        with pytest.raises(FileNotFoundError, match="No image files"):
            src.in_dist_frame()

    def test_extensions_filter(self, tmp_path):
        """Non-image files should be ignored."""
        _make_dummy_pngs(tmp_path, n=2, size=32)
        (tmp_path / "notes.txt").write_text("not an image")
        (tmp_path / "data.csv").write_text("1,2,3")
        src = DiskFrameSource(
            in_dist_dir=str(tmp_path),
            img_size=32,
            rng=np.random.default_rng(0),
        )
        src._ensure_in_dist()
        assert len(src._in_dist_frames) == 2


# ---------------------------------------------------------------------------
# RobomimicFrameSource with mock HDF5
# ---------------------------------------------------------------------------

class TestRobomimicFrameSource:
    def test_load_from_mock_hdf5(self, tmp_path):
        """Create a tiny mock HDF5 and verify loading works."""
        h5py = pytest.importorskip("h5py")
        from frame_sources import RobomimicFrameSource

        hdf5_path = tmp_path / "test.hdf5"
        n_frames = 5
        img_shape = (n_frames, 48, 48, 3)
        imgs = np.random.randint(0, 255, img_shape, dtype=np.uint8)

        with h5py.File(hdf5_path, "w") as f:
            demo = f.create_group("data/demo_0/obs")
            demo.create_dataset("agentview_image", data=imgs)
            demo.create_dataset("robot0_eye_in_hand_image",
                                data=np.random.randint(0, 255, img_shape, dtype=np.uint8))

        src = RobomimicFrameSource(
            hdf5_path=str(hdf5_path),
            img_size=32,
            max_frames=10,
            rng=np.random.default_rng(0),
        )
        frame = src.in_dist_frame()
        assert frame.shape == (32, 32, 3)
        assert frame.dtype == np.float32

        mild = src.mild_ood_frame()
        assert mild.shape == (32, 32, 3)

    def test_missing_camera_raises(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        from frame_sources import RobomimicFrameSource

        hdf5_path = tmp_path / "test.hdf5"
        with h5py.File(hdf5_path, "w") as f:
            demo = f.create_group("data/demo_0/obs")
            demo.create_dataset("agentview_image",
                                data=np.zeros((2, 32, 32, 3), dtype=np.uint8))

        src = RobomimicFrameSource(
            hdf5_path=str(hdf5_path),
            camera="nonexistent",
            img_size=32,
            rng=np.random.default_rng(0),
        )
        with pytest.raises(KeyError, match="nonexistent"):
            src.in_dist_frame()


# ---------------------------------------------------------------------------
# ImageNetOODSource — import fallback
# ---------------------------------------------------------------------------

def test_imagenet_source_raises_without_datasets():
    """ImageNetOODSource should fail gracefully if datasets not installed."""
    from frame_sources import ImageNetOODSource
    src = ImageNetOODSource(max_frames=5, rng=np.random.default_rng(0))
    # We can't test the actual loading without network, but we can verify
    # that the object is constructed and in_dist_frame raises NotImplementedError
    with pytest.raises(NotImplementedError):
        src.in_dist_frame()
    with pytest.raises(NotImplementedError):
        src.mild_ood_frame()


# ---------------------------------------------------------------------------
# LeRobotFrameSource — construction only (no network)
# ---------------------------------------------------------------------------

def test_lerobot_source_construction():
    """Verify LeRobotFrameSource can be constructed without network."""
    from frame_sources import LeRobotFrameSource
    src = LeRobotFrameSource(
        in_dist_repo="lerobot/pusht_image",
        mild_ood_repo="lerobot/utokyo_xarm_pick_and_place",
        img_size=64,
        max_frames=10,
        rng=np.random.default_rng(0),
    )
    assert src.in_dist_repo == "lerobot/pusht_image"
    assert src._in_dist_frames is None  # lazy, not loaded yet
