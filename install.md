Both pyproject.toml and uv.lock are already committed to the repo, confirming this project's init.cmd was run once (by whoever set it up) and its metadata/lockfile checked in. Here's what's actually needed on this specific Windows 11 machine.

Current state on this machine

┌─────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────┐
│        Check        │                                          Result                                           │
├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ pyproject.toml /    │ Present, tracked in git — dependency metadata already exists                              │
│ uv.lock             │                                                                                           │
├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ .venv               │ Missing — nothing has been synced locally yet                                             │
├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ uv                  │ Not installed (not found in PowerShell or Git Bash)                                       │
├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ System Python       │ 3.14.0 (project needs ≥3.12 — uv manages its own interpreter, so this isn't blocking)     │
├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ GPU                 │ NVIDIA RTX 2070 SUPER, driver 591.86 (8GB VRAM — below the "comfortable 24GB" note in the │
│                     │  README, fine for 50M dev-scale work and CPU-verified tests)                              │
└─────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────┘

So despite init.cmd having been run at some point (against this repo, on some machine), this checkout still needs its local environment created — that step isn't captured by git.

check cuda version:
nvidia-smi
if nvidia needs driver update
check how much vram

MiniFrontier requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/). On Windows, install `uv`
with any one of these methods.

Official PowerShell installer:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Windows Package Manager:

```powershell
winget install astral-sh.uv
```
best way is:  winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements


Or, when Python and `pip` are already available:

```powershell
python -m pip install --user --upgrade uv
```
(winget already handles it)

Open a new terminal if the installer changed `PATH`, then verify the prerequisites:

```powershell
python --version
uv --version
```

Windows users can then initialize the checkout with the idempotent bootstrap:

```bat
init.cmd cu130
```


$machinePath = [System.Environment]::GetEnvironmentVariable("Path","Machine")
$userPath = [System.Environment]::GetEnvironmentVariable("Path","User")
$env:Path = $machinePath + ";" + $userPath
& .\init.cmd cu130 

Available PyTorch backends are `cpu`, `cu126`, `cu130`, and `cu132`; `cu130` is the project default. Choose the wheel supported by the installed NVIDIA driver, or use `cpu` for correctness work. Backends are mutually exclusive locked extras, so `init.cmd cpu` on a Dev Box and `init.cmd cu130` on an RTX machine use the same project metadata safely. The default sync installs the core and development groups. Add `--all-groups` when evaluation and plotting dependencies are needed. The script creates missing directories and package markers but never overwrites an existing file.





Plan

1. Install uv (pick one):
   - winget install astral-sh.uv (simplest on Win11), or
   - powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   - Open a new terminal afterward so PATH picks it up.
   
2. Verify prerequisites: python --version and uv --version.

3. Bootstrap with the GPU backend: run init.cmd cu130 from the repo root.
   - Since pyproject.toml/uv.lock already exist, this skips project re-init and just runs uv sync --extra cu130 --group dev, downloading the pinned Python 3.12 + PyTorch cu130 wheel + deps into a fresh .venv.
   - init.cmd then self-verifies: prints Python/PyTorch/CUDA/BF16 info, checks import minifrontier, and runs pytest if tests exist.
   - The 2070 Super (Turing, CC 7.5) is compatible with cu130 wheels under driver 591.86.
   
4. Run the full check loop per README:
uv run --extra cu130 ruff check .
uv run --extra cu130 pytest


uv run --extra cu130 python -m pytest tests/test_prepare_data.py -q
uv run --extra cu130 python -m pytest -q
uv python pin 3.12
uv sync --extra cu130 --group dev

uv run --extra cu130 python -c "import sys; print('Python:', sys.version); import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); print('BF16 supported:', torch.cuda.is_bf16_supported())")

should be similar to this:
Python: 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
PyTorch: 2.13.0+cu130
CUDA available: True
CUDA runtime: 13.0
GPU: NVIDIA GeForce RTX 2070 SUPER
BF16 supported: True
 
5. Read tasks/backlog.md to pick the next MF-NNN item per AGENTS.md's workflow, and check its listed dependencies are Done before starting.

Optional: if you want a lightweight CPU-only correctness environment too (e.g., to match the CI/"Dev Box" gate mentioned in the README), init.cmd cpu can be run separately — the cpu/cu130 extras are mutually exclusive per sync but both are valid against the same lockfile.