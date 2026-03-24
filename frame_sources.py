"""
Real Data Frame Sources
========================
Frame source classes for loading real robotics datasets and using them
with the OOD detection system.

Classes:
    LeRobotFrameSource   — loads from any LeRobot-compatible HuggingFace dataset
    ImageNetOODSource    — hard OOD baseline from ImageNet-50 subset
    RobomimicFrameSource — loads from robomimic HDF5 files
    DiskFrameSource      — loads images from directories on disk
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize a HWC float32 frame to size×size using PIL."""
    from PIL import Image
    h, w = frame.shape[:2]
    if h == size and w == size:
        return frame
    img = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8))
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


class ImageNetOODSource:
    """
    Hard OOD source from the ImageNet-50 subset on HuggingFace.

    Provides only hard_ood_frame(). Used internally by other sources.

    Parameters
    ----------
    max_frames : int
        Maximum number of frames to cache in memory.
    img_size : int
        Resize images to img_size × img_size.
    rng : np.random.Generator or None
        Random generator for frame sampling.
    """

    def __init__(self, max_frames: int = 500, img_size: int = 224,
                 rng: Optional[np.random.Generator] = None):
        self.max_frames = max_frames
        self.img_size = img_size
        self.rng = rng or np.random.default_rng()
        self._frames: Optional[list] = None

    def _ensure_loaded(self):
        if self._frames is not None:
            return
        from datasets import load_dataset

        print(f"[ImageNetOOD] loading up to {self.max_frames} frames from "
              f"Elriggs/imagenet-50-subset ...")
        ds = load_dataset("Elriggs/imagenet-50-subset", split="train",
                          streaming=True)
        self._frames = []
        for item in ds:
            if len(self._frames) >= self.max_frames:
                break
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            frame = np.asarray(img, dtype=np.float32) / 255.0
            frame = _resize_frame(frame, self.img_size)
            self._frames.append(frame)
        print(f"[ImageNetOOD] loaded {len(self._frames)} frames")

    def hard_ood_frame(self) -> np.ndarray:
        self._ensure_loaded()
        idx = self.rng.integers(0, len(self._frames))
        return self._frames[idx]

    def in_dist_frame(self) -> np.ndarray:
        raise NotImplementedError(
            "ImageNetOODSource only provides hard_ood_frame(). "
            "Use a robot dataset source for in_dist_frame().")

    def mild_ood_frame(self) -> np.ndarray:
        raise NotImplementedError(
            "ImageNetOODSource only provides hard_ood_frame(). "
            "Use a robot dataset source for mild_ood_frame().")


class LeRobotFrameSource:
    """
    Loads frames from LeRobot-compatible HuggingFace datasets.

    OOD split strategy:
    - in_dist: frames from in_dist_repo
    - mild_ood: frames from mild_ood_repo (different robot/task)
    - hard_ood: ImageNet-50 images

    Parameters
    ----------
    in_dist_repo : str
        LeRobot dataset repo for in-distribution frames.
    mild_ood_repo : str
        LeRobot dataset repo for mild OOD frames.
    hard_ood_repo : str or None
        LeRobot dataset repo for hard OOD frames. None = use ImageNet-50.
    image_key : str or None
        Key for image observations. None = auto-detect first image key.
    img_size : int
        Resize all frames to img_size × img_size.
    max_frames : int
        Max frames to cache per split.
    rng : np.random.Generator or None
    """

    def __init__(
        self,
        in_dist_repo: str = "lerobot/pusht_image",
        mild_ood_repo: str = "lerobot/utokyo_xarm_pick_and_place",
        hard_ood_repo: Optional[str] = None,
        image_key: Optional[str] = None,
        img_size: int = 224,
        max_frames: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ):
        self.in_dist_repo = in_dist_repo
        self.mild_ood_repo = mild_ood_repo
        self.hard_ood_repo = hard_ood_repo
        self.image_key = image_key
        self.img_size = img_size
        self.max_frames = max_frames
        self.rng = rng or np.random.default_rng()

        self._in_dist_frames: Optional[list] = None
        self._mild_ood_frames: Optional[list] = None
        self._hard_ood_frames: Optional[list] = None
        if hard_ood_repo is None:
            self._imagenet = ImageNetOODSource(
                max_frames=max_frames, img_size=img_size, rng=self.rng)
        else:
            self._imagenet = None

    def _load_repo(self, repo: str, label: str) -> list:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import torch

        print(f"[LeRobot] loading {label} from {repo} ...")
        # Load first few episodes to stay within max_frames
        ds = LeRobotDataset(repo, episodes=[0])
        n_episodes = ds.meta.total_episodes

        # Find image key
        image_key = self.image_key
        if image_key is None:
            sample = ds[0]
            image_keys = [k for k in sample.keys() if "image" in k.lower()]
            if not image_keys:
                raise ValueError(f"No image keys found in {repo}. "
                                 f"Available keys: {list(sample.keys())}")
            image_key = image_keys[0]
            print(f"[LeRobot] auto-detected image key: {image_key}")

        # Load episodes until we have enough frames
        frames = []
        ep_idx = 0
        while len(frames) < self.max_frames and ep_idx < n_episodes:
            ds = LeRobotDataset(repo, episodes=[ep_idx])
            for i in range(len(ds)):
                if len(frames) >= self.max_frames:
                    break
                item = ds[i]
                img_tensor = item[image_key]  # CHW float32 torch tensor
                if isinstance(img_tensor, torch.Tensor):
                    img = img_tensor.permute(1, 2, 0).numpy()  # HWC
                else:
                    img = np.asarray(img_tensor, dtype=np.float32)
                img = np.clip(img, 0, 1).astype(np.float32)
                img = _resize_frame(img, self.img_size)
                frames.append(img)
            ep_idx += 1

        print(f"[LeRobot] loaded {len(frames)} {label} frames "
              f"from {ep_idx} episodes")
        return frames

    def _ensure_in_dist(self):
        if self._in_dist_frames is None:
            self._in_dist_frames = self._load_repo(
                self.in_dist_repo, "in-dist")

    def _ensure_mild_ood(self):
        if self._mild_ood_frames is None:
            self._mild_ood_frames = self._load_repo(
                self.mild_ood_repo, "mild-OOD")

    def in_dist_frame(self) -> np.ndarray:
        self._ensure_in_dist()
        idx = self.rng.integers(0, len(self._in_dist_frames))
        return self._in_dist_frames[idx]

    def mild_ood_frame(self) -> np.ndarray:
        self._ensure_mild_ood()
        idx = self.rng.integers(0, len(self._mild_ood_frames))
        return self._mild_ood_frames[idx]

    def _ensure_hard_ood(self):
        if self._hard_ood_frames is None and self.hard_ood_repo is not None:
            self._hard_ood_frames = self._load_repo(
                self.hard_ood_repo, "hard-OOD")

    def hard_ood_frame(self) -> np.ndarray:
        if self.hard_ood_repo is not None:
            self._ensure_hard_ood()
            idx = self.rng.integers(0, len(self._hard_ood_frames))
            return self._hard_ood_frames[idx]
        return self._imagenet.hard_ood_frame()


class RobomimicFrameSource:
    """
    Loads camera observations from robomimic HDF5 datasets.

    OOD split strategy:
    - in_dist: specified camera from the HDF5 file
    - mild_ood: different camera view from the same HDF5
    - hard_ood: ImageNet-50 images

    Parameters
    ----------
    hdf5_path : str
        Path to robomimic HDF5 dataset file.
    camera : str
        Camera name for in-distribution frames.
    ood_camera : str
        Camera name for mild OOD frames (different viewpoint).
    img_size : int
        Resize all frames to img_size × img_size.
    max_frames : int
        Max frames to cache per split.
    rng : np.random.Generator or None
    """

    def __init__(
        self,
        hdf5_path: str,
        camera: str = "agentview",
        ood_camera: str = "robot0_eye_in_hand",
        img_size: int = 224,
        max_frames: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ):
        self.hdf5_path = hdf5_path
        self.camera = camera
        self.ood_camera = ood_camera
        self.img_size = img_size
        self.max_frames = max_frames
        self.rng = rng or np.random.default_rng()

        self._in_dist_frames: Optional[list] = None
        self._mild_ood_frames: Optional[list] = None
        self._imagenet = ImageNetOODSource(
            max_frames=max_frames, img_size=img_size, rng=self.rng)

    def _load_camera(self, camera: str, label: str) -> list:
        import h5py

        print(f"[Robomimic] loading {label} from {self.hdf5_path} "
              f"camera={camera} ...")
        frames = []
        with h5py.File(self.hdf5_path, "r") as f:
            data = f["data"]
            demo_keys = sorted(
                [k for k in data.keys() if k.startswith("demo_")],
                key=lambda k: int(k.split("_")[1]),
            )
            obs_key = f"{camera}_image"
            for dk in demo_keys:
                if len(frames) >= self.max_frames:
                    break
                obs = data[dk]["obs"]
                if obs_key not in obs:
                    available = list(obs.keys())
                    raise KeyError(
                        f"Camera '{camera}' not found (looked for '{obs_key}'). "
                        f"Available obs keys: {available}")
                imgs = obs[obs_key][:]  # (T, H, W, C) uint8
                for img in imgs:
                    if len(frames) >= self.max_frames:
                        break
                    frame = np.asarray(img, dtype=np.float32) / 255.0
                    frame = _resize_frame(frame, self.img_size)
                    frames.append(frame)

        print(f"[Robomimic] loaded {len(frames)} {label} frames")
        return frames

    def _ensure_in_dist(self):
        if self._in_dist_frames is None:
            self._in_dist_frames = self._load_camera(
                self.camera, "in-dist")

    def _ensure_mild_ood(self):
        if self._mild_ood_frames is None:
            self._mild_ood_frames = self._load_camera(
                self.ood_camera, "mild-OOD")

    def in_dist_frame(self) -> np.ndarray:
        self._ensure_in_dist()
        idx = self.rng.integers(0, len(self._in_dist_frames))
        return self._in_dist_frames[idx]

    def mild_ood_frame(self) -> np.ndarray:
        self._ensure_mild_ood()
        idx = self.rng.integers(0, len(self._mild_ood_frames))
        return self._mild_ood_frames[idx]

    def hard_ood_frame(self) -> np.ndarray:
        return self._imagenet.hard_ood_frame()


class DiskFrameSource:
    """
    Loads images from directories on disk (PNG/JPG).

    Parameters
    ----------
    in_dist_dir : str
        Directory with in-distribution images.
    mild_ood_dir : str or None
        Directory with mild OOD images. If None, mild_ood_frame() falls back
        to in_dist with random color jitter.
    hard_ood_dir : str or None
        Directory with hard OOD images. If None, uses ImageNet-50 subset.
    img_size : int
        Resize all frames to img_size × img_size.
    rng : np.random.Generator or None
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        in_dist_dir: str,
        mild_ood_dir: Optional[str] = None,
        hard_ood_dir: Optional[str] = None,
        img_size: int = 224,
        rng: Optional[np.random.Generator] = None,
    ):
        self.in_dist_dir = Path(in_dist_dir)
        self.mild_ood_dir = Path(mild_ood_dir) if mild_ood_dir else None
        self.hard_ood_dir = Path(hard_ood_dir) if hard_ood_dir else None
        self.img_size = img_size
        self.rng = rng or np.random.default_rng()

        self._in_dist_frames: Optional[list] = None
        self._mild_ood_frames: Optional[list] = None
        self._hard_ood_frames: Optional[list] = None
        if self.hard_ood_dir is None:
            self._imagenet = ImageNetOODSource(
                img_size=img_size, rng=self.rng)
        else:
            self._imagenet = None

    def _load_dir(self, directory: Path, label: str) -> list:
        from PIL import Image

        print(f"[Disk] loading {label} from {directory} ...")
        files = sorted(
            f for f in directory.iterdir()
            if f.suffix.lower() in self.EXTENSIONS
        )
        if not files:
            raise FileNotFoundError(
                f"No image files found in {directory}")

        frames = []
        for f in files:
            img = Image.open(f).convert("RGB")
            frame = np.asarray(img, dtype=np.float32) / 255.0
            frame = _resize_frame(frame, self.img_size)
            frames.append(frame)

        print(f"[Disk] loaded {len(frames)} {label} frames")
        return frames

    def _ensure_in_dist(self):
        if self._in_dist_frames is None:
            self._in_dist_frames = self._load_dir(
                self.in_dist_dir, "in-dist")

    def _ensure_mild_ood(self):
        if self._mild_ood_frames is not None:
            return
        if self.mild_ood_dir is not None:
            self._mild_ood_frames = self._load_dir(
                self.mild_ood_dir, "mild-OOD")
        else:
            # Fallback: color-jittered versions of in-dist
            self._ensure_in_dist()
            self._mild_ood_frames = []
            for frame in self._in_dist_frames:
                jittered = frame.copy()
                jittered = jittered * self.rng.uniform(0.7, 1.3, (1, 1, 3)).astype(np.float32)
                jittered = np.clip(jittered, 0, 1)
                self._mild_ood_frames.append(jittered)

    def _ensure_hard_ood(self):
        if self._hard_ood_frames is not None:
            return
        if self.hard_ood_dir is not None:
            self._hard_ood_frames = self._load_dir(
                self.hard_ood_dir, "hard-OOD")

    def in_dist_frame(self) -> np.ndarray:
        self._ensure_in_dist()
        idx = self.rng.integers(0, len(self._in_dist_frames))
        return self._in_dist_frames[idx]

    def mild_ood_frame(self) -> np.ndarray:
        self._ensure_mild_ood()
        idx = self.rng.integers(0, len(self._mild_ood_frames))
        return self._mild_ood_frames[idx]

    def hard_ood_frame(self) -> np.ndarray:
        if self._imagenet is not None:
            return self._imagenet.hard_ood_frame()
        self._ensure_hard_ood()
        idx = self.rng.integers(0, len(self._hard_ood_frames))
        return self._hard_ood_frames[idx]
