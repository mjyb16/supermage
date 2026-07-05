# Installation

## Requirements

- Python ≥ 3.8
- [PyTorch](https://pytorch.org) (a CUDA build is strongly recommended —
  everything runs on CPU, but GPU acceleration is where SuperMAGE shines)
- [caskade](https://github.com/Ciela-Institute/caskade) (parameter-graph
  framework all models are built on)
- [caustics](https://github.com/Ciela-Institute/caustics) (only needed for
  lensed-cube modeling)
- [viscube](https://github.com/mjyb16/viscube) (gridding of interferometric
  data that the visibility simulators are designed to match)
- astropy, numpy, scipy, graphviz, safetensors

## Install from source

```bash
git clone https://github.com/mjyb16/supermage.git
cd supermage
pip install -e .
```

The editable (`-e`) install is recommended while the package is under
active development.

If you want a specific CUDA build of PyTorch, install torch *first*
following the [official instructions](https://pytorch.org/get-started/locally/),
then run the `pip install -e .` above.

## Verify the installation

```python
import torch, supermage
print(supermage.__version__, "| CUDA available:", torch.cuda.is_available())
```

## Building the documentation locally

```bash
pip install -r docs/requirements.txt
jupyter-book build docs/source/
```

The rendered book lands in `docs/source/_build/html/`. The tutorial
notebooks are stored executed (`execute_notebooks: 'off'`), so building the
docs does not require a GPU.
