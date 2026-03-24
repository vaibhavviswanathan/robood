"""
VLA OOD Detection — Live Demo Script
=====================================
Ties together:
  • Frame source  — robosuite sim OR synthetic fallback (no GPU needed)
  • Encoder       — DINOv2 (real) OR lightweight CNN proxy (no GPU needed)
  • Detector      — VLAOODDetector (Mahalanobis)
  • Visualisation — live matplotlib dashboard updating every frame

Run modes
---------
    # Full sim + real encoder (needs robosuite + torch + GPU)
    python vla_ood_demo.py --mode sim --encoder dinov2

    # Synthetic frames + lightweight encoder (no dependencies beyond numpy/matplotlib)
    python vla_ood_demo.py --mode synthetic --encoder proxy

    # Synthetic frames + DINOv2 (torch required, no robosuite)
    python vla_ood_demo.py --mode synthetic --encoder dinov2

Demo story
----------
  Phase 1 (frames 0–49)   : in-distribution — normal robot workspace
  Phase 2 (frames 50–74)  : mild OOD — slightly novel background / lighting
  Phase 3 (frames 75–99)  : hard OOD — completely novel object / scene
The score trace and per-frame indicators update live so the audience can see
the detector reacting in real time.
"""

from __future__ import annotations
import argparse
import time
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation
from pathlib import Path

# ---------------------------------------------------------------------------
# Import detector from the core module
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from vla_ood_detector import VLAOODDetector, OODResult


# ===========================================================================
# FRAME SOURCES
# ===========================================================================

class SyntheticFrameSource:
    """
    Generates realistic-looking synthetic robot workspace frames using numpy.
    No robosuite or GPU required.

    Frame anatomy
    -------------
    • Table surface  — beige/grey rectangle (lower half)
    • Robot arm      — simple geometric overlay (always present)
    • Task object    — coloured blob (in-dist: red cube, OOD: random shape/colour)
    • Background     — solid or gradient (in-dist: grey wall, OOD: random texture)
    • Distractors    — extra objects placed at random (hard OOD only)
    """

    H, W = 224, 224

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    # --- public API ---------------------------------------------------------

    def in_dist_frame(self) -> np.ndarray:
        """Normal robot workspace: grey wall, beige table, red cube."""
        img = self._background_wall(hue_shift=0.0)
        img = self._draw_table(img, brightness=0.72)
        img = self._draw_arm(img)
        img = self._draw_cube(img, color=(0.80, 0.15, 0.10),
                              cx=self.W // 2, cy=int(self.H * 0.62), size=24)
        img = self._add_noise(img, sigma=0.012)
        return img

    def mild_ood_frame(self) -> np.ndarray:
        """Slightly different: warmer lighting, table brightness shifted."""
        shift = self.rng.uniform(0.08, 0.18)
        img = self._background_wall(hue_shift=shift)
        img = self._draw_table(img, brightness=self.rng.uniform(0.55, 0.65))
        img = self._draw_arm(img)
        # Same cube but colour drifted
        r = 0.80 + self.rng.uniform(-0.15, 0.15)
        img = self._draw_cube(img, color=(np.clip(r, 0, 1), 0.20, 0.10),
                              cx=self.W // 2, cy=int(self.H * 0.62), size=24)
        img = self._add_noise(img, sigma=0.025)
        return img

    def hard_ood_frame(self) -> np.ndarray:
        """Completely novel: random background, random objects, no table."""
        # Random vivid background
        bg_col = self.rng.uniform(0.1, 0.9, 3).astype(np.float32)
        img = np.ones((self.H, self.W, 3), dtype=np.float32) * bg_col
        # Random large distractor blobs
        for _ in range(self.rng.integers(3, 7)):
            col  = self.rng.uniform(0.0, 1.0, 3)
            cx   = self.rng.integers(30, self.W - 30)
            cy   = self.rng.integers(30, self.H - 30)
            size = self.rng.integers(20, 55)
            img  = self._draw_ellipse(img, col, cx, cy, size,
                                      int(size * self.rng.uniform(0.4, 1.5)))
        img = self._add_noise(img, sigma=0.04)
        return np.clip(img, 0, 1)

    # --- drawing primitives --------------------------------------------------

    def _background_wall(self, hue_shift=0.0) -> np.ndarray:
        base = np.array([0.55 + hue_shift * 0.3,
                         0.55 + hue_shift * 0.1,
                         0.58 - hue_shift * 0.1], dtype=np.float32)
        img = np.ones((self.H, self.W, 3), dtype=np.float32) * base
        # Subtle vertical gradient
        grad = np.linspace(0.0, 0.06, self.H)[:, None, None]
        img += grad
        return np.clip(img, 0, 1)

    def _draw_table(self, img, brightness=0.72) -> np.ndarray:
        img = img.copy()
        table_top = int(self.H * 0.55)
        # Table surface
        img[table_top:, :] = brightness
        # Table edge highlight
        img[table_top:table_top + 3, :] = np.clip(brightness + 0.12, 0, 1)
        return img

    def _draw_arm(self, img) -> np.ndarray:
        img = img.copy()
        arm_col = np.array([0.88, 0.88, 0.88], dtype=np.float32)
        # Upper arm segment
        self._draw_rect(img, arm_col,
                        x=self.W // 2 - 8, y=int(self.H * 0.25),
                        w=16, h=int(self.H * 0.28))
        # Forearm (angled via shear approximation)
        self._draw_rect(img, arm_col,
                        x=self.W // 2 - 6, y=int(self.H * 0.50),
                        w=12, h=int(self.H * 0.14))
        # Gripper fingers
        self._draw_rect(img, np.array([0.70, 0.70, 0.70]),
                        x=self.W // 2 - 14, y=int(self.H * 0.60),
                        w=10, h=8)
        self._draw_rect(img, np.array([0.70, 0.70, 0.70]),
                        x=self.W // 2 + 4, y=int(self.H * 0.60),
                        w=10, h=8)
        return img

    def _draw_cube(self, img, color, cx, cy, size) -> np.ndarray:
        img = img.copy()
        col = np.array(color, dtype=np.float32)
        half = size // 2
        # Front face
        self._draw_rect(img, col,
                        x=cx - half, y=cy - half, w=size, h=size)
        # Top face (lighter)
        top = np.clip(col + 0.18, 0, 1)
        pts_x = [cx - half, cx + half, cx + half + 8, cx - half + 8]
        pts_y = [cy - half, cy - half, cy - half - 8, cy - half - 8]
        self._draw_quad(img, top, pts_x, pts_y)
        # Right face (darker)
        side = np.clip(col - 0.18, 0, 1)
        pts_x2 = [cx + half, cx + half + 8, cx + half + 8, cx + half]
        pts_y2 = [cy - half, cy - half - 8, cy + half - 8, cy + half]
        self._draw_quad(img, side, pts_x2, pts_y2)
        return img

    def _draw_ellipse(self, img, color, cx, cy, rx, ry) -> np.ndarray:
        img = img.copy()
        col = np.array(color, dtype=np.float32)
        ys, xs = np.ogrid[:self.H, :self.W]
        mask = ((xs - cx) / max(rx, 1)) ** 2 + ((ys - cy) / max(ry, 1)) ** 2 <= 1
        img[mask] = col
        return img

    @staticmethod
    def _draw_rect(img, color, x, y, w, h):
        H, W = img.shape[:2]
        x0, x1 = max(0, x), min(W, x + w)
        y0, y1 = max(0, y), min(H, y + h)
        img[y0:y1, x0:x1] = color

    @staticmethod
    def _draw_quad(img, color, xs, ys):
        H, W = img.shape[:2]
        x0, x1 = max(0, int(min(xs))), min(W, int(max(xs)) + 1)
        y0, y1 = max(0, int(min(ys))), min(H, int(max(ys)) + 1)
        img[y0:y1, x0:x1] = color

    @staticmethod
    def _add_noise(img, sigma=0.015):
        return np.clip(img + np.random.normal(0, sigma, img.shape), 0, 1)


class RobosuiteFrameSource:
    """
    Wraps robosuite to produce real rendered frames.
    Uses DomainRandomizationWrapper for OOD generation.
    Falls back gracefully if robosuite isn't installed.
    """

    def __init__(self, task="Lift", robot="Panda", img_size=224):
        try:
            import robosuite as suite
            from robosuite.wrappers import DomainRandomizationWrapper

            base_cfg = dict(
                robots=robot,
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_names="agentview",
                camera_heights=img_size,
                camera_widths=img_size,
            )
            self._env_clean = suite.make(task, **base_cfg)
            self._env_ood = DomainRandomizationWrapper(
                suite.make(task, **base_cfg),
                randomize_color=True,
                randomize_lighting=True,
                randomize_camera=True,
                randomize_every_n_steps=1,
            )
            self._available = True
            print(f"[RobosuiteSource] loaded task={task} robot={robot}")
        except ImportError:
            print("[RobosuiteSource] robosuite not found — falling back to synthetic")
            self._available = False
            self._fallback = SyntheticFrameSource(np.random.default_rng(0))

    def _obs_to_frame(self, obs) -> np.ndarray:
        frame = obs["agentview_image"]   # uint8 HxWx3
        return frame.astype(np.float32) / 255.0

    def in_dist_frame(self) -> np.ndarray:
        if not self._available:
            return self._fallback.in_dist_frame()
        obs, _ = self._env_clean.reset()
        return self._obs_to_frame(obs)

    def mild_ood_frame(self) -> np.ndarray:
        if not self._available:
            return self._fallback.mild_ood_frame()
        obs, _ = self._env_ood.reset()
        return self._obs_to_frame(obs)

    def hard_ood_frame(self) -> np.ndarray:
        if not self._available:
            return self._fallback.hard_ood_frame()
        obs, _ = self._env_ood.reset()
        return self._obs_to_frame(obs)


# ===========================================================================
# ENCODERS
# ===========================================================================

def make_proxy_encoder(embedding_dim: int = 128):
    """
    Lightweight CNN-style encoder using only numpy.
    Computes multi-scale colour/texture histograms — surprisingly effective
    for visual OOD and works without torch or GPU.
    """
    def encoder_fn(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        H, W, _ = img.shape

        features = []

        # 1. Global colour histogram (RGB, 16 bins each → 48)
        for c in range(3):
            hist, _ = np.histogram(img[:, :, c], bins=16, range=(0, 1))
            features.append(hist / (H * W))

        # 2. Spatial grid mean colour (4×4 grid × 3 channels → 48)
        gh, gw = 4, 4
        for i in range(gh):
            for j in range(gw):
                patch = img[i * H // gh:(i + 1) * H // gh,
                            j * W // gw:(j + 1) * W // gw]
                features.append(patch.mean(axis=(0, 1)))

        # 3. Gradient magnitude histogram (16 bins → 16)
        gray = img.mean(axis=2)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        hist_g, _ = np.histogram(grad_mag, bins=16, range=(0, 1))
        features.append(hist_g / (H * W))

        # 4. Brightness + contrast scalars (2)
        features.append(np.array([gray.mean(), gray.std()]))

        vec = np.concatenate(features).astype(np.float32)
        # Pad or trim to embedding_dim
        if len(vec) < embedding_dim:
            vec = np.pad(vec, (0, embedding_dim - len(vec)))
        else:
            vec = vec[:embedding_dim]
        return vec

    return encoder_fn


def make_dinov2_encoder(device: str = "cpu"):
    """
    Real DINOv2 ViT-S/14 encoder via torch.hub.
    Requires: pip install torch torchvision
    """
    import torch
    import torchvision.transforms as T

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                           pretrained=True)
    model.eval().to(device)

    transform = T.Compose([
        T.ToTensor(),
        T.Resize(224, antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    def encoder_fn(image: np.ndarray) -> np.ndarray:
        from PIL import Image as PILImage
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        pil = PILImage.fromarray(image)
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            z = model(tensor).squeeze(0).cpu().numpy()
        return z.astype(np.float32)

    return encoder_fn


# ===========================================================================
# DEMO DASHBOARD
# ===========================================================================

class OODDemoDashboard:
    """
    Live matplotlib dashboard with:
      ┌────────────────┬──────────────────────────────┐
      │  Current frame │   OOD score trace             │
      │                │                               │
      │  Status badge  │   Threshold line              │
      ├────────────────┴──────────────────────────────┤
      │  Phase indicator bar                           │
      └────────────────────────────────────────────────┘
    """

    PHASES = [
        (0,  50, "In-distribution",  "#2ecc71"),
        (50, 75, "Mild OOD",         "#f39c12"),
        (75, 100, "Hard OOD",        "#e74c3c"),
    ]
    N_FRAMES = 100

    def __init__(self, detector: VLAOODDetector, frame_source,
                 interval_ms: int = 120):
        self.detector = detector
        self.source   = frame_source
        self.interval = interval_ms

        # History buffers
        self.scores: list[float] = []
        self.flags:  list[bool]  = []
        self.frames: list[np.ndarray] = []

        self._build_figure()

    # -------------------------------------------------------------------------

    def _build_figure(self):
        matplotlib.rcParams.update({
            "font.family":    "monospace",
            "text.color":     "#e8e8e8",
            "axes.labelcolor": "#e8e8e8",
            "xtick.color":    "#888",
            "ytick.color":    "#888",
            "axes.edgecolor": "#444",
            "figure.facecolor": "#111",
            "axes.facecolor":   "#1a1a1a",
            "grid.color":       "#2a2a2a",
        })

        self.fig = plt.figure(figsize=(14, 6), facecolor="#111")
        self.fig.canvas.manager.set_window_title("VLA OOD Detection — Live Demo")

        gs = gridspec.GridSpec(
            2, 2,
            figure=self.fig,
            width_ratios=[1, 2.2],
            height_ratios=[5, 1],
            hspace=0.08,
            wspace=0.22,
            left=0.06, right=0.97,
            top=0.93, bottom=0.08,
        )

        # Left: camera frame
        self.ax_frame = self.fig.add_subplot(gs[0, 0])
        self.ax_frame.set_title("Robot camera", fontsize=10,
                                color="#aaa", pad=6)
        self.ax_frame.axis("off")
        self._im = self.ax_frame.imshow(
            np.zeros((224, 224, 3), dtype=np.float32),
            vmin=0, vmax=1, aspect="equal"
        )
        # Status badge (text overlay)
        self._badge = self.ax_frame.text(
            0.5, 0.04, "WAITING",
            transform=self.ax_frame.transAxes,
            ha="center", va="bottom", fontsize=13, fontweight="bold",
            color="#111",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#888",
                      edgecolor="none", alpha=0.92),
        )

        # Right top: score trace
        self.ax_score = self.fig.add_subplot(gs[0, 1])
        method_label = self.detector.method.capitalize()
        self.ax_score.set_title(f"{method_label} OOD score",
                                fontsize=10, color="#aaa", pad=6)
        self.ax_score.set_xlim(0, self.N_FRAMES)
        self.ax_score.set_xlabel("frame", fontsize=9)
        self.ax_score.set_ylabel("score (d²)", fontsize=9)
        self.ax_score.grid(True, linewidth=0.5)
        self.ax_score.set_yscale("symlog", linthresh=100)

        # Threshold line (drawn once, updated)
        tau = self.detector._threshold
        self._thresh_line = self.ax_score.axhline(
            tau, color="#e74c3c", linewidth=1.2,
            linestyle="--", label=f"τ = {tau:.1f}"
        )
        self._thresh_label = self.ax_score.text(
            self.N_FRAMES * 0.98, tau * 1.03,
            f"τ={tau:.1f}", color="#e74c3c",
            fontsize=8, ha="right", va="bottom"
        )

        # Phase background bands
        for (x0, x1, label, col) in self.PHASES:
            self.ax_score.axvspan(x0, x1, alpha=0.07, color=col, linewidth=0)
            self.ax_score.text(
                (x0 + x1) / 2, 0.97, label,
                transform=self.ax_score.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.5,
                color=col, alpha=0.85,
            )

        # Score line + OOD scatter
        self._score_line, = self.ax_score.plot(
            [], [], color="#5dade2", linewidth=1.4, zorder=3
        )
        self._ood_scatter = self.ax_score.scatter(
            [], [], color="#e74c3c", s=28, zorder=4, label="OOD flagged"
        )
        self._ok_scatter = self.ax_score.scatter(
            [], [], color="#2ecc71", s=18, zorder=4, label="in-dist"
        )
        self.ax_score.legend(loc="upper left", fontsize=8,
                             framealpha=0.3, edgecolor="#444")

        # Bottom: phase progress bar
        self.ax_bar = self.fig.add_subplot(gs[1, :])
        self.ax_bar.set_xlim(0, self.N_FRAMES)
        self.ax_bar.set_ylim(0, 1)
        self.ax_bar.axis("off")
        for (x0, x1, label, col) in self.PHASES:
            self.ax_bar.barh(0.5, x1 - x0, left=x0, height=0.6,
                             color=col, alpha=0.25, linewidth=0)
            self.ax_bar.text(
                (x0 + x1) / 2, 0.5, label,
                ha="center", va="center", fontsize=8, color=col
            )
        self._progress = self.ax_bar.axvline(
            0, color="#fff", linewidth=1.5, alpha=0.6
        )

        plt.tight_layout(pad=1.2)

    # -------------------------------------------------------------------------

    def _get_frame(self, idx: int) -> np.ndarray:
        if idx < 50:
            return self.source.in_dist_frame()
        elif idx < 75:
            return self.source.mild_ood_frame()
        else:
            return self.source.hard_ood_frame()

    def _phase_color(self, idx: int) -> str:
        for (x0, x1, _, col) in self.PHASES:
            if x0 <= idx < x1:
                return col
        return "#888"

    def _update(self, frame_idx: int):
        img   = self._get_frame(frame_idx)
        result = self.detector.score(img)

        self.scores.append(result.score)
        self.flags.append(result.is_ood)
        self.frames.append(img)

        xs = list(range(len(self.scores)))

        # ---- camera frame ----
        self._im.set_data(np.clip(img, 0, 1))

        # ---- status badge ----
        if result.is_ood:
            self._badge.set_text("⚠  OOD DETECTED — HALT")
            self._badge.get_bbox_patch().set_facecolor("#e74c3c")
        else:
            self._badge.set_text("✓  IN-DISTRIBUTION")
            self._badge.get_bbox_patch().set_facecolor("#2ecc71")

        # ---- score trace ----
        self._score_line.set_data(xs, self.scores)

        ood_xs = [i for i, f in enumerate(self.flags) if f]
        ok_xs  = [i for i, f in enumerate(self.flags) if not f]
        ood_ys = [self.scores[i] for i in ood_xs]
        ok_ys  = [self.scores[i] for i in ok_xs]

        if ood_xs:
            self._ood_scatter.set_offsets(np.c_[ood_xs, ood_ys])
        if ok_xs:
            self._ok_scatter.set_offsets(np.c_[ok_xs, ok_ys])

        # Auto-scale y with headroom
        if self.scores:
            ymax = max(self.scores) * 1.25
            ymin = min(self.scores) * 0.75
            self.ax_score.set_ylim(max(0, ymin), max(ymax, self.detector._threshold * 1.5))

        # ---- progress bar ----
        self._progress.set_xdata([frame_idx + 1, frame_idx + 1])

        # Print to terminal too
        phase = next(l for (x0, x1, l, _) in self.PHASES
                     if x0 <= frame_idx < x1)
        flag_str = "OOD ⚠" if result.is_ood else "OK  ✓"
        print(f"  frame {frame_idx:03d} | {phase:<20s} | "
              f"score={result.score:8.2f} | τ={result.threshold:.2f} | {flag_str}")

        return (self._im, self._badge, self._score_line,
                self._ood_scatter, self._ok_scatter, self._progress)

    def run(self):
        self._anim = FuncAnimation(
            self.fig,
            self._update,
            frames=self.N_FRAMES,
            interval=self.interval,
            blit=False,
            repeat=False,
        )
        plt.show()

    def save_gif(self, path: str = "ood_demo.gif", fps: int = 8):
        """Save demo as a GIF for slide decks."""
        print(f"[Dashboard] rendering GIF → {path}  (may take a moment...)")
        anim = FuncAnimation(
            self.fig, self._update,
            frames=self.N_FRAMES, interval=self.interval,
            blit=False, repeat=False,
        )
        anim.save(path, writer="pillow", fps=fps)
        print(f"[Dashboard] saved {path}")


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="VLA OOD Detection Live Demo")
    p.add_argument("--mode", choices=["synthetic", "sim", "lerobot", "robomimic", "disk"],
                   default="synthetic",
                   help="Frame source: synthetic, sim, lerobot, robomimic, or disk")
    p.add_argument("--encoder", choices=["proxy", "dinov2"],
                   default="proxy",
                   help="Encoder: proxy (no deps) or dinov2 (needs torch)")
    p.add_argument("--method", choices=["mahalanobis", "knn"],
                   default="mahalanobis",
                   help="OOD scoring method: mahalanobis or knn")
    p.add_argument("--n-fit", type=int, default=300,
                   help="Number of in-dist frames to fit the detector on")
    p.add_argument("--pca", type=int, default=32,
                   help="PCA components (set 0 to disable)")
    p.add_argument("--threshold-pct", type=float, default=95.0,
                   help="Percentile for threshold (default: 95)")
    p.add_argument("--interval", type=int, default=120,
                   help="Milliseconds between frames in live plot")
    p.add_argument("--save-gif", action="store_true",
                   help="Save the demo as ood_demo.gif instead of showing live")
    p.add_argument("--save-png", action="store_true",
                   help="Save a static sample frames comparison to sample_frames.png")
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


def main():
    args = parse_args()
    rng = np.random.default_rng(42)

    print("\n" + "=" * 60)
    print("  VLA OOD Detection — Live Demo")
    print("=" * 60)
    print(f"  mode={args.mode}  encoder={args.encoder}  method={args.method}  "
          f"n_fit={args.n_fit}  pca={args.pca}")
    print()

    # ------------------------------------------------------------------
    # 1. Frame source
    # ------------------------------------------------------------------
    if args.mode == "sim":
        source = RobosuiteFrameSource()
    elif args.mode == "lerobot":
        from frame_sources import LeRobotFrameSource
        source = LeRobotFrameSource(
            in_dist_repo=args.dataset_repo,
            mild_ood_repo=args.mild_ood_repo,
            img_size=224,
            max_frames=args.max_frames,
            rng=rng,
        )
    elif args.mode == "robomimic":
        from frame_sources import RobomimicFrameSource
        if not args.dataset_path:
            print("[Error] --dataset-path required for robomimic mode")
            sys.exit(1)
        source = RobomimicFrameSource(
            hdf5_path=args.dataset_path,
            img_size=224,
            max_frames=args.max_frames,
            rng=rng,
        )
    elif args.mode == "disk":
        from frame_sources import DiskFrameSource
        if not args.dataset_path:
            print("[Error] --dataset-path required for disk mode")
            sys.exit(1)
        source = DiskFrameSource(
            in_dist_dir=args.dataset_path,
            img_size=224,
            rng=rng,
        )
    else:
        source = SyntheticFrameSource(rng)

    # ------------------------------------------------------------------
    # 2. Encoder
    # ------------------------------------------------------------------
    if args.encoder == "dinov2":
        try:
            print("[Encoder] loading DINOv2 ViT-S/14 ...")
            encoder_fn = make_dinov2_encoder(device="cpu")
            print("[Encoder] DINOv2 ready")
        except Exception as e:
            print(f"[Encoder] DINOv2 failed ({e}), falling back to proxy")
            encoder_fn = make_proxy_encoder()
    else:
        encoder_fn = make_proxy_encoder()
        print("[Encoder] proxy encoder ready (numpy only)")

    # ------------------------------------------------------------------
    # 3. Fit detector on in-dist frames
    # ------------------------------------------------------------------
    print(f"\n[Detector] collecting {args.n_fit} in-dist frames for fitting ...")
    fit_images = [source.in_dist_frame() for _ in range(args.n_fit)]

    detector = VLAOODDetector(
        encoder_fn=encoder_fn,
        method=args.method,
        threshold_percentile=args.threshold_pct,
        pca_components=args.pca if args.pca > 0 else None,
    )
    detector.fit(fit_images, verbose=True)

    # Quick sanity check before the demo
    print("\n[Sanity] scoring 5 in-dist and 5 hard-OOD frames:")
    for _ in range(5):
        r = detector.score(source.in_dist_frame())
        print(f"  in-dist  score={r.score:8.2f}  is_ood={r.is_ood}")
    for _ in range(5):
        r = detector.score(source.hard_ood_frame())
        print(f"  hard-OOD score={r.score:8.2f}  is_ood={r.is_ood}")

    # ------------------------------------------------------------------
    # 4. Save static comparison or launch live dashboard
    # ------------------------------------------------------------------
    if args.save_png:
        print("\n[Demo] saving static sample frames → sample_frames.png")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (gen, label) in zip(axes, [
            (source.in_dist_frame, "In-Distribution"),
            (source.mild_ood_frame, "Mild OOD"),
            (source.hard_ood_frame, "Hard OOD"),
        ]):
            frame = gen()
            r = detector.score(frame)
            ax.imshow(np.clip(frame, 0, 1))
            color = "#2ecc71" if not r.is_ood else "#e74c3c"
            status = "OK" if not r.is_ood else "OOD"
            ax.set_title(f"{label}\nscore={r.score:.1f} [{status}]",
                         color=color, fontsize=11)
            ax.axis("off")
        fig.suptitle("VLA OOD Detection — Sample Frames", fontsize=13)
        fig.tight_layout()
        fig.savefig("sample_frames.png", dpi=120)
        plt.close(fig)
        print("[Demo] saved sample_frames.png")
        return

    print("\n[Demo] starting live dashboard ...\n")
    print(f"  {'frame':<7} | {'phase':<20s} | {'score':>10} | {'τ':>8} | status")
    print("  " + "-" * 60)

    dashboard = OODDemoDashboard(
        detector=detector,
        frame_source=source,
        interval_ms=args.interval,
    )

    if args.save_gif:
        # Pre-run all frames then save
        for i in range(OODDemoDashboard.N_FRAMES):
            dashboard._update(i)
        dashboard.save_gif("ood_demo.gif")
    else:
        dashboard.run()


if __name__ == "__main__":
    main()
