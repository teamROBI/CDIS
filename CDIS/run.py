# "Cross-Dimensional Instance Segmentation: Utilizing 2D Models for Generalizable 3D Solutions"
from __future__ import annotations

import os
import sys
from pathlib import Path
import argparse
import multiprocessing as mp

from omegaconf import OmegaConf
from natsort import natsorted

from utils.util import manage_temp_directory
from CDIS.segment_instances import seg_inst

REPO_ROOT = Path(__file__).resolve().parents[1]
CROPFORMER_DIR = REPO_ROOT / "libs" / "detectron2" / "projects" / "CropFormer"


def _import_cropformer():
    """Import detectron2 + CropFormer lazily.

    These are only needed to GENERATE 2D masks (``data.data_all_ready: false``). The documented
    default runs on precomputed masks, and importing them eagerly would make detectron2,
    CropFormer and mmcv hard requirements for every user. Keeping the import here means the
    default path needs none of them.
    """
    # detectron2's project finder references importlib.abc without importing it, which is no
    # longer implicitly available on Python 3.10+. Import it explicitly first.
    import importlib.abc  # noqa: F401
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config

    if CROPFORMER_DIR.is_dir() and str(CROPFORMER_DIR) not in sys.path:
        sys.path.append(str(CROPFORMER_DIR))
    from mask2former import add_maskformer2_config
    from demo_cropformer.predictor import VisualizationDemo

    return get_cfg, add_deeplab_config, add_maskformer2_config, VisualizationDemo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CDIS")
    p.add_argument("--config_path", type=str, required=True, help="Path to YAML config")
    p.add_argument("--part", type=int, default=0, help="Current subprocess index (0-based)")
    p.add_argument("--subprocess_num", type=int, default=1, help="Total number of subprocesses")
    return p.parse_args()


def load_config() -> "OmegaConf":
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)
    cfg.part = int(args.part)
    cfg.subprocess_num = max(1, int(args.subprocess_num))

    # Basic validation
    total_parts = cfg.subprocess_num
    if not (0 <= cfg.part < total_parts):
        raise ValueError(f"--part must be in [0,{total_parts-1}], got {cfg.part}")

    print("✅ Config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    return cfg


def setup_cropformer(setting, get_cfg, add_deeplab_config, add_maskformer2_config) -> "CfgNode":
    cf = get_cfg()
    add_deeplab_config(cf)
    add_maskformer2_config(cf)
    cf.merge_from_file(setting["config-file"])
    cf.merge_from_list(["MODEL.WEIGHTS", setting["model-weights"]])
    cf.freeze()
    return cf


def get_mask_generator(cfg):
    if cfg.mask_model != "cropformer":
        raise ValueError(f"Unsupported mask_model '{cfg.mask_model}'; only 'cropformer' is supported.")

    if cfg.data.data_all_ready:
        # Precomputed masks: the 2D model is never used, so it is never imported or loaded.
        print("[INFO] Using precomputed CropFormer masks (2D model not loaded)")
        return None

    print("[INFO] Loading CropFormer to generate 2D masks")
    get_cfg, add_deeplab_config, add_maskformer2_config, VisualizationDemo = _import_cropformer()
    # set spawn once at entrypoint ideally, but keep here if needed for demo
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    return VisualizationDemo(
        setup_cropformer(cfg.vfm_config.cropformer_config,
                         get_cfg, add_deeplab_config, add_maskformer2_config)
    )


def process_scene(cfg, scene_name: str, mask_generator) -> None:
    save_path = Path(cfg.exp.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    scene_pth = save_path / f"{scene_name}.pth"

    if scene_pth.exists():
        print(f"[SKIP] {scene_pth.name} already exists.")
        return

    print(f"[PROCESSING] Scene: {scene_name}")
    seg_inst(cfg, scene_name, mask_generator)

    # temp dir configurable? keeps current default
    manage_temp_directory(os.path.join(cfg.data.data_root, "temp_dense_pcd"))


def main() -> None:
    # Set spawn once, safest for CUDA/Detectron
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    cfg = load_config()

    # Scenes
    with open(cfg.data.scene_name_path) as f:
        all_scenes = natsorted([s for s in (ln.strip() for ln in f) if s])

    Path(cfg.exp.save_path).mkdir(parents=True, exist_ok=True)
    mask_generator = get_mask_generator(cfg)

    total = len(all_scenes)
    part_len = total // cfg.subprocess_num
    start = cfg.part * part_len
    end = total if cfg.part == cfg.subprocess_num - 1 else start + part_len
    target = all_scenes[start:end]

    if not getattr(cfg, "full_process", True):
        chosen = set(getattr(cfg, "chosen_scene", []))
        target = [s for s in target if s in chosen]

    print("✅ Starting CDIS processing...")
    print(f"Total scenes: {total}")
    print(f"[INFO] Subprocess range: {start} - {end} (count={len(target)})")

    failed = []
    for scene_name in target:
        try:
            process_scene(cfg, scene_name, mask_generator)
        except Exception as exc:  # keep going so one bad scene does not abort the run
            import traceback
            print(f"[ERROR] Scene {scene_name} failed: {exc}")
            traceback.print_exc()
            failed.append(scene_name)
    if failed:
        print(f"[WARN] {len(failed)} scene(s) failed: {failed}")


if __name__ == "__main__":
    main()
