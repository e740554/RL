# Reinforcement Learning (RL) Repo

This repository contains a small RL setup with a custom environment and a Soft Actor-Critic (SAC) training script. It is intended as a minimalist starting point for experimentation, iteration, and extensions.

## Overview
- Custom environment in `env_energyhub.py` (an Energy Hub–style environment).
- Training entrypoint in `train_sac.py` (SAC baseline).
- TensorBoard logs stored under `tb/`.

## Quick Start
1) Create and activate a virtual environment (recommended):
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - Unix/macOS:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

2) Install dependencies. If you don’t have a `requirements.txt`, install the basics used by many SAC setups:
   ```bash
   pip install numpy torch gymnasium tensorboard
   # Add any additional packages used by your code
   ```

3) Run training:
   ```bash
   python train_sac.py
   ```

4) Monitor with TensorBoard:
   ```bash
   tensorboard --logdir tb
   ```
   Then open the URL printed in your terminal.

## Repository Layout
- `env_energyhub.py` — custom EnergyHub environment implementation.
- `train_sac.py` — SAC training script and experiment entrypoint.
- `tb/` — TensorBoard logs (ignored by Git).
- `.venv/` — local virtual environment (ignored by Git).

## Notes
- GPU is optional; if available and supported by your PyTorch install, training will typically be faster.
- For reproducibility, consider fixing seeds (NumPy, PyTorch, environment) in your training script.
- Keep large artifacts (models, logs) out of Git; use the `tb/`, `checkpoints/`, or external storage.

## License
No license has been specified. Add a license if you plan to share or distribute this code.

