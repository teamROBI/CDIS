# Installation

Tested with **Python 3.10, PyTorch 1.11.0, CUDA 11.3** on Linux with an NVIDIA GPU.

CDIS has two installation levels:

| You want to… | You need |
|---|---|
| **Run the pipeline on precomputed 2D masks** (the default, `data.data_all_ready: true`) | Step 1 only |
| **Generate the 2D masks yourself** (`data.data_all_ready: false`) | Steps 1 + 2 |

Most users only need step 1. detectron2/CropFormer are imported lazily, so they are not required
unless you are actually generating masks.

## 1. Environment (uv)

We use [uv](https://docs.astral.sh/uv/). Install it if you do not have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the repository root:

```bash
uv sync            # creates .venv/ and installs the locked dependencies
```

That is the whole environment. `uv.lock` pins every dependency, so the install is reproducible.
Run commands with `uv run`, e.g.:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

or activate the venv once with `source .venv/bin/activate` and use `python` directly.

PyTorch 1.11 + CUDA 11.3 wheels come from the `pytorch-cu113` index declared in `pyproject.toml`.
If your driver needs a different CUDA build, change that index URL and the `torch` / `torchvision`
pins, then re-run `uv lock`.

## 2. Optional: 2D mask generation (detectron2 + CropFormer)

Only needed for `data.data_all_ready: false`. These compile CUDA extensions against your local
toolkit, so they cannot be locked and must be built manually.

You need an `nvcc` matching PyTorch's CUDA (11.3) on the `PATH`; a system CUDA 12.x will not build
cleanly against CUDA 11.3 wheels:

```bash
export CUDA_HOME=/usr/local/cuda-11.3
export PATH="$CUDA_HOME/bin:$PATH"
# Set the arch for your GPU: RTX 8000=7.5, RTX 3090=8.6, A100=8.0, H100=9.0
# (https://developer.nvidia.com/cuda-gpus)
export TORCH_CUDA_ARCH_LIST="8.6"
```

```bash
source .venv/bin/activate

# `uv sync` creates a venv WITHOUT pip; detectron2's editable build and openmim both need it.
# setuptools must stay <81: torch 1.11's cpp_extension imports pkg_resources, which newer
# setuptools no longer ships, and every CUDA extension build below would fail.
uv pip install pip "setuptools<81" wheel

# detectron2 v0.6 (the last release supporting torch 1.11)
git clone https://github.com/facebookresearch/detectron2.git libs/detectron2
( cd libs/detectron2 && git checkout v0.6 )
uv pip install --no-build-isolation -e libs/detectron2

# CropFormer is vendored in this repo and builds as a detectron2 project
cp -r libs/CropFormer libs/detectron2/projects/
( cd libs/detectron2/projects/CropFormer/entity_api/PythonAPI && uv pip install pycocotools==2.0.7 && make )
( cd libs/detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/ops && sh make.sh )

# CropFormer's backbones (timm) and its dataset registry (mmcv)
uv pip install timm openmim && mim install mmcv

# detectron2/mmcv may pull in numpy 2.x, which breaks numba and the torch 1.11 numpy bridge
uv pip install numpy==1.24.4
```

Verify the optional stack, using the same lazy import CDIS itself performs (`mask2former` lives
under `libs/detectron2/projects/CropFormer`, which CDIS adds to `sys.path` at call time, so it is
not importable on its own):

```bash
PYTHONPATH=./ python -c "from CDIS.run import _import_cropformer; _import_cropformer(); print('CropFormer stack OK')"
```

## 3. Checkpoints

Download the CropFormer checkpoint into `weights/` — see [../weights/README.md](../weights/README.md).
Only needed for step 2.

## 4. Data

Prepare the dataset as described in [DATA.md](DATA.md). All inputs and caches live under a single
root (`data.data_root`, default `data`), so one symlink is enough:

```bash
ln -s /path/to/storage/CDIS/data data
```

`output/` is created automatically; symlink it too if you want predictions on another disk.

## Troubleshooting

- **`AttributeError: module 'importlib' has no attribute 'abc'`** — detectron2 v0.6 references
  `importlib.abc` without importing it, which Python 3.10+ no longer does implicitly. CDIS imports
  it explicitly before detectron2; if you hit this from your own script, `import importlib.abc`
  first.
- **`ModuleNotFoundError: mask2former`** — the `cp -r libs/CropFormer libs/detectron2/projects/`
  step in section 2 has not been run.
- **numba/numpy errors** — something pulled in numpy 2.x; re-pin `numpy==1.24.4`.
