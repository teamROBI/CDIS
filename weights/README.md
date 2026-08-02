# Model checkpoints

The pretrained 2D-segmentation checkpoints are **not** committed to the repository (they are
large and gitignored). Download the checkpoint into this `weights/` directory before running the
pipeline. Note it is only needed when generating 2D masks (`data.data_all_ready: false`); with
precomputed masks the 2D model is never loaded.

| File | Model | Size | Source |
|------|-------|------|--------|
| `CropFormer_hornet_3x_03823a.pth` | CropFormer (HorNet, 3x) — the 2D class-agnostic mask model used in the paper | ~849 MB | [Adobe_EntitySeg on Hugging Face](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/tree/main/CropFormer_model/Entity_Segmentation/CropFormer_hornet_3x) |

## Download

```bash
# CropFormer HorNet 3x (from the Adobe_EntitySeg Hugging Face repo)
wget -O weights/CropFormer_hornet_3x_03823a.pth \
  "https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/resolve/main/CropFormer_model/Entity_Segmentation/CropFormer_hornet_3x/CropFormer_hornet_3x_03823a.pth"
```

The default path expected by the code (`configs/scannet200.yaml`) is:
- `weights/CropFormer_hornet_3x_03823a.pth` (`vfm_config.cropformer_config.model-weights`)
