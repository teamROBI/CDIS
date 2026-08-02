#!/usr/bin/env python
"""
Class-agnostic 3D instance-segmentation evaluator for CDIS.

Wraps the vendored ScanNet AP protocol (evaluation/eval_semantic_instance.py,
util.py, util_3d.py) and evaluates CDIS predictions with class-agnostic
AP / AP50 / AP25 over IoU thresholds 0.5:0.95:0.05 (+0.25).

Prediction format (per scene): a 1-D int array of length = #scene points, where
each entry is a predicted instance id and -1 means background/unassigned. Each
unique id >= 0 becomes one predicted instance mask (confidence = 1.0). Files may
be .npy or .pth (numpy array or torch tensor). The scene name is taken from the
file basename with the extension and an optional leading "pred_" stripped.

Ground truth: {gt_dir}/{scene}.txt with one integer per point encoded as
semantic_label*1000 + instance_index (instance_index 0 => no instance / bg).
Only values >= 1000 count as ground-truth instances (ScanNet convention, kept
identical to the OVMap implementation).

Usage:
  # full validation set (all pred files in a dir vs the GT dir)
  PYTHONPATH=./ python evaluation/run_class_agnostic_eval.py \
      --pred_dir  /path/to/preds \
      --gt_dir    /data/.../instance_gt/validation

  # single scene (quick sanity check)
  PYTHONPATH=./ python evaluation/run_class_agnostic_eval.py \
      --pred_file /path/to/pred_scene0011_00.npy \
      --gt_dir    /data/.../instance_gt/validation
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_semantic_instance as evi


def load_pointwise_pred(path):
    """Load a per-point prediction array (.npy or .pth) as a 1-D int64 numpy array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in (".pth", ".pt"):
        import torch
        arr = torch.load(path, map_location="cpu")
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        arr = np.asarray(arr)
    else:
        raise ValueError("Unsupported prediction file extension: {}".format(path))
    arr = np.asarray(arr).reshape(-1).astype(np.int64)
    return arr


def pointwise_to_pred_dict(pred_ids):
    """Convert a per-point instance-id array into the {pred_masks,pred_scores,pred_classes}
    format expected by eval_semantic_instance.evaluate().

    pred_masks:   N x M binary, one column per predicted instance
    pred_scores:  M, all 1.0 (single confidence)
    pred_classes: M, all 1  (ignored in class-agnostic mode)
    """
    pred_ids = np.asarray(pred_ids).reshape(-1)
    n_points = pred_ids.shape[0]
    inst_ids = np.unique(pred_ids)
    inst_ids = inst_ids[inst_ids >= 0]  # drop background (-1)

    m = len(inst_ids)
    pred_masks = np.zeros((n_points, m), dtype=np.uint8)
    for i, iid in enumerate(inst_ids):
        pred_masks[:, i] = (pred_ids == iid)

    pred_scores = np.ones(m, dtype=np.float32)
    pred_classes = np.ones(m, dtype=np.int64)  # class-agnostic -> label ignored
    return {
        "pred_masks": pred_masks,
        "pred_scores": pred_scores,
        "pred_classes": pred_classes,
    }


def scene_name_from_file(path):
    base = os.path.splitext(os.path.basename(path))[0]
    if base.startswith("pred_"):
        base = base[len("pred_"):]
    return base


def collect_pred_files(pred_dir):
    files = []
    for f in sorted(os.listdir(pred_dir)):
        if os.path.splitext(f)[1].lower() in (".npy", ".pth", ".pt"):
            files.append(os.path.join(pred_dir, f))
    return files


def build_preds(pred_files, gt_dir):
    preds = {}
    for pf in pred_files:
        scene = scene_name_from_file(pf)
        gt_file = os.path.join(gt_dir, scene + ".txt")
        if not os.path.isfile(gt_file):
            print("[WARN] no GT for scene '{}' (looked for {}); skipping {}".format(
                scene, gt_file, pf))
            continue
        pred_ids = load_pointwise_pred(pf)
        n_gt = len(open(gt_file).read().splitlines())
        if len(pred_ids) != n_gt:
            print("[WARN] scene {}: pred has {} points but GT has {}; skipping".format(
                scene, len(pred_ids), n_gt))
            continue
        n_inst = int((np.unique(pred_ids) >= 0).sum())
        print("[INFO] scene {}: {} points, {} predicted instances".format(
            scene, len(pred_ids), n_inst))
        preds[scene] = pointwise_to_pred_dict(pred_ids)
    return preds


def main():
    parser = argparse.ArgumentParser(
        description="Class-agnostic 3D instance-seg AP/AP50/AP25 evaluation (ScanNet protocol).")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred_dir", type=str, default=None,
                   help="Directory of per-point prediction .npy/.pth files (full-val mode).")
    g.add_argument("--pred_file", type=str, default=None,
                   help="Single per-point prediction .npy/.pth file (single-scene mode).")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory of GT .txt files ({scene}.txt).")
    parser.add_argument("--scene", type=str, default=None,
                        help="Override scene name (single-scene mode); "
                             "otherwise inferred from --pred_file basename.")
    parser.add_argument("--output_file", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "class_agnostic_eval_result.txt"),
                        help="Where to write the per-class/summary result file.")
    args = parser.parse_args()

    if args.pred_file is not None:
        scene = args.scene if args.scene else scene_name_from_file(args.pred_file)
        # single-scene mode: reuse build_preds by faking a one-file list, but honor --scene
        gt_file = os.path.join(args.gt_dir, scene + ".txt")
        if not os.path.isfile(gt_file):
            print("[ERROR] GT file not found: {}".format(gt_file))
            sys.exit(1)
        pred_ids = load_pointwise_pred(args.pred_file)
        n_gt = len(open(gt_file).read().splitlines())
        if len(pred_ids) != n_gt:
            print("[ERROR] scene {}: pred has {} points but GT has {}".format(
                scene, len(pred_ids), n_gt))
            sys.exit(1)
        n_inst = int((np.unique(pred_ids) >= 0).sum())
        print("[INFO] scene {}: {} points, {} predicted instances".format(
            scene, len(pred_ids), n_inst))
        preds = {scene: pointwise_to_pred_dict(pred_ids)}
    else:
        pred_files = collect_pred_files(args.pred_dir)
        if not pred_files:
            print("[ERROR] no .npy/.pth prediction files found in {}".format(args.pred_dir))
            sys.exit(1)
        preds = build_preds(pred_files, args.gt_dir)

    if not preds:
        print("[ERROR] no evaluable scenes.")
        sys.exit(1)

    avgs, ar_avgs, rc_avgs, pcdc_avgs = evi.evaluate(
        preds, args.gt_dir, output_file=args.output_file, class_agnostic=True)

    # Concise class-agnostic summary table
    ap = avgs["all_ap"]
    ap50 = avgs["all_ap_50%"]
    ap25 = avgs["all_ap_25%"]
    print("")
    print("=" * 52)
    print("CLASS-AGNOSTIC 3D INSTANCE SEGMENTATION  ({} scene(s))".format(len(preds)))
    print("=" * 52)
    print("{:<12}{:>12}{:>12}{:>12}".format("", "AP", "AP50", "AP25"))
    print("{:<12}{:>12.4f}{:>12.4f}{:>12.4f}".format("class-agnostic", ap, ap50, ap25))
    print("=" * 52)
    print("Result file written to: {}".format(args.output_file))


if __name__ == "__main__":
    main()
