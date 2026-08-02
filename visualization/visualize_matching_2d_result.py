import argparse
import os
import re
import glob
import random
import sys
import numpy as np
import cv2

# =============================
# Utilities
# =============================

class Colorizer:
    """Keeps a consistent ID->RGB color map (no globals)."""
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.color_map = {-1: [0, 0, 0]}

    def reset_colors(self):
        """Reset all assigned colors (except background) for fresh random ones."""
        self.color_map = {-1: [0, 0, 0]}
        if self.seed is not None:
            random.seed(self.seed)

    def get_color(self, id_):
        if id_ not in self.color_map:
            while True:
                color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                # avoid pure black and pure red
                if color != [0, 0, 0] and color != [255, 0, 0]:
                    self.color_map[id_] = color
                    break
        return self.color_map[id_]

    def colorize_mask_with_ids(self, mask: np.ndarray) -> np.ndarray:
        """RGB visualization with white ID text."""
        h, w = mask.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        unique_ids = np.unique(mask)
        for id_ in unique_ids:
            if id_ == 0:
                continue  # background
            color = self.get_color(int(id_))
            vis[mask == id_] = color
            ys, xs = np.where(mask == id_)
            if xs.size == 0 or ys.size == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(
                vis, str(int(id_)), (cx, cy),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA
            )
        return vis


def natural_key(s: str):
    """Sort helper that splits strings into text and integer chunks."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def load_masks_from_npz(npz_path: str) -> np.ndarray:
    """Load (N,H,W) integer masks from NPZ (uses 'masks' if present, else first array)."""
    with np.load(npz_path, allow_pickle=True) as data:
        arr = data['masks'] if 'masks' in data.files else data[data.files[0]]
    if arr.ndim != 3:
        raise ValueError(f"Expected (N,H,W) in {npz_path}, got shape {arr.shape}")
    return arr.astype(np.int32)


def make_strip(colorizer: Colorizer, masks: np.ndarray, start: int, window: int) -> np.ndarray:
    """Build a side-by-side strip from masks[start : start+window]."""
    N = masks.shape[0]
    end = min(start + window, N)
    if start >= end:
        return None
    frames = [colorizer.colorize_mask_with_ids(masks[i]) for i in range(start, end)]
    return np.concatenate(frames, axis=1)


# =============================
# Main (interactive sliding window preview)
# =============================

def main():
    parser = argparse.ArgumentParser(description="Preview 2D instance matching from *_masks.npz with a sliding window (OpenCV only).")
    parser.add_argument("input_dir", type=str, help="Directory containing {scene_name}_masks.npz files")
    parser.add_argument("--pattern", type=str, default="*_masks.npz", help="Glob pattern (default: '*_masks.npz')")
    parser.add_argument("--window", type=int, default=5, help="Sliding window size (default: 10)")
    parser.add_argument("--scale", type=float, default=1.0, help="Resize factor for display (e.g., 0.5)")
    parser.add_argument("--keep-colormap", action="store_true",
                        help="Keep the same color map across scenes (default: reset per scene)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for color generation (optional)")
    args = parser.parse_args()

    inp = os.path.abspath(args.input_dir)
    files = sorted(glob.glob(os.path.join(inp, args.pattern)), key=natural_key)
    if not files:
        print(f"No files matched {args.pattern} under {inp}")
        return

    window_name = "Matching Preview"

    persistent_colorizer = Colorizer(seed=args.seed) if args.keep_colormap else None

    print("Controls: SPACE=advance, LEFT/RIGHT=step, R=reset colors, Q/ESC=quit")

    for f in reversed(files):
        colorizer = persistent_colorizer or Colorizer(seed=args.seed)

        base = os.path.basename(f)
        scene = re.sub(r"_masks\.npz$", "", base)

        try:
            masks = load_masks_from_npz(f)
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")
            continue

        N = masks.shape[0]
        print(f"\nVisualizing scene: {scene} ({N} frames)")

        start = 0

        while True:
            strip_rgb = make_strip(colorizer, masks, start, args.window)
            if strip_rgb is None:
                break

            end = min(start + args.window, N)
            sys.stdout.write(f"\r{scene}  frames [{start}..{end-1}] / {N}")
            sys.stdout.flush()

            strip_bgr = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2BGR)
            if args.scale != 1.0:
                h, w = strip_bgr.shape[:2]
                strip_bgr = cv2.resize(strip_bgr, (int(w * args.scale), int(h * args.scale)),
                                       interpolation=cv2.INTER_NEAREST)

            cv2.imshow(window_name, strip_bgr)

            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord('q')):  # ESC or 'q'
                cv2.destroyAllWindows()
                return
            elif key == 32:  # SPACE
                if start < N - 1:
                    start += 1
                else:
                    break
            elif key == 81:  # LEFT arrow
                if start > 0:
                    start -= 1
            elif key == 83:  # RIGHT arrow
                if start < N - 1:
                    start += 1
            elif key in (ord('r'), ord('R')):  # Reset colors
                colorizer.reset_colors()
            else:
                pass

        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

    cv2.destroyAllWindows()
    print("\nDone.")

if __name__ == "__main__":
    main()
