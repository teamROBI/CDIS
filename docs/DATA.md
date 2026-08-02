# Data preparation

CDIS reads datasets from a `data/` directory and writes predictions to `output/` at the
repository root. Both are untracked by git — create them as local directories or symlinks to a
large-storage location. The dataset root read by the pipeline is set by `data.data_root` in
`configs/scannet200.yaml`.

```bash
ln -s /path/to/storage/CDIS/data   data
ln -s /path/to/storage/CDIS/output output
```

## ScanNet / ScanNet200

Request access to and download ScanNet from the official repository:
**https://github.com/ScanNet/ScanNet** (you must agree to the ScanNet Terms of Use). The
validation split used in the paper (312 scenes) is listed in
`data/scannetv2/scannet_preprocess/meta_data/scannetv2_val.txt`.

### Expected layout

```
<data_root>/scannetv2/
├── input/
│   ├── scannetv2_images/val/<scene>/{color,depth,pose,intrinsics}/     # posed RGB-D frames  -> data.scans_2d_path
│   ├── scannetv2_pcds/val/<scene>.ply                                  # scene point cloud    -> data.pcd_path
│   └── mesh_segmentation/0.01_20/<scene>_vh_clean_2.0.010000.segs.json # 3D superpoints       -> data.spp_path
├── process_saved/
│   ├── 2d_seg/cropformer/<scene>/<frame>.png                           # precomputed 2D masks -> exp.mask2d_output
│   └── depth_from_pc/<scene>/<frame>.png                               # synthetic depth      -> data.depths.synthetic_depth_path
└── instance_gt/validation/<scene>.txt                                  # evaluation GT        -> scripts/run_eval.sh
```

The evaluation ground truth uses the openmask3d instance-GT format: one integer per point,
`semantic_label * 1000 + instance_id`. `scripts/run_eval.sh` defaults to
`data/scannetv2/instance_gt/validation`; pass a different location as its second argument
(or `--gt_dir`) if you keep it elsewhere:

```bash
bash scripts/run_eval.sh <PRED_DIR> <GT_DIR>
```

### Preprocessing steps

1. **Posed RGB-D frames** (from raw `.sens` scans): extract colour, depth, poses and intrinsics
   with the [ScanNet toolkit](https://github.com/ScanNet/ScanNet) (`SensReader`) and export the
   scene meshes as point clouds. This produces the `scannetv2_images/val` and `scannetv2_pcds/val`
   trees shown above. [Pointcept](https://github.com/Pointcept/Pointcept) provides equivalent
   ScanNet preprocessing scripts.

2. **Mesh over-segmentation** (the 3D superpoints): the default config uses **kThresh = 0.01**,
   which is the segmentation ScanNet already ships as
   `<scene>_vh_clean_2.0.010000.segs.json` — no extra tool needed. Copy those files to
   `input/mesh_segmentation/0.01_20/`.

   Superpoint granularity matters: they are the atomic unit of the 3D merge, so a superpoint that
   spans two objects can never be split into separate instances. Coarser over-segmentations measure
   clearly worse (see the granularity comparison in the reproduction notes). Coarser levels can be
   generated with ScanNet's
   [`segmentator`](https://github.com/ScanNet/ScanNet/tree/master/Segmentator) and selected by
   pointing `data.spp_path` at e.g. `mesh_segmentation/0.05_20/...0.050000.segs.json`.

3. **Synthetic / supplemented depth** (`depth_type: sup_depth`): ScanNet's sensor depth has holes,
   so CDIS supplements it with depth rendered from the scene point cloud, cached under
   `process_saved/depth_from_pc/`. These are generated automatically on first use: run the pipeline
   once and any missing per-frame depth is rendered and cached (see
   the depth handling in `utils/CDIS_util.py`). To skip this entirely, set
   `depth_type: raw_depth`.

4. **2D masks**: precomputed into `process_saved/2d_seg/cropformer/` as 16-bit PNG label maps.
   Run the pipeline once with `data.data_all_ready: false` — CDIS then loads CropFormer and caches
   the masks as it goes. With `data.data_all_ready: true` (default) the pipeline reads the cached
   masks and never loads the 2D model (so detectron2/CropFormer are not needed at all).

