# Reproducible Environment

## Verified training environment

The Xinjiang 2021 E0/E1 development and training pipeline is verified with:

| Component | Version |
| --- | --- |
| Operating system | Windows |
| Python | 3.11.14 |
| PyTorch | 2.10.0+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| NumPy | 2.3.5 |
| PyYAML | 6.0.3 |
| Matplotlib | 3.10.8 |
| pytest | 8.3.4 |

The local interpreter used for the formal run is:

```text
D:\Anaconda3\envs\cawa\python.exe
```

This absolute path is machine-local evidence only. It must not be embedded in shared model or experiment configuration.

## Installation

Create and activate a Python 3.11 environment. Install the PyTorch CUDA 12.8 build from the official PyTorch wheel index, then install the remaining pinned packages:

```powershell
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

For CPU-only contract tests, install the appropriate PyTorch build for the host instead. Formal E0/E1 training requires CUDA and validates `torch.cuda.is_available()` before starting.

## Verification

```powershell
python -c "import torch, numpy, yaml, matplotlib, pytest; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(numpy.__version__, yaml.__version__, matplotlib.__version__, pytest.__version__)"
python -m pytest -q
```

The experiment record must capture the actual Python, PyTorch, CUDA, GPU, and package versions again at run time; this file records the approved development baseline, not a substitute for per-run provenance.
