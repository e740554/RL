# AGENTS.md

This file provides guidance for agents (and contributors) working in this repository. Its scope applies to the entire repo.

## Project Context
- Purpose: Reinforcement Learning experiments with a custom EnergyHub-style environment and a SAC training script.
- Key files:
  - `env_energyhub.py`: environment implementation.
  - `train_sac.py`: training entrypoint and experiment logic.
  - `tb/`: TensorBoard logs (ignored by Git).

## Run & Validate
- Use a local virtual environment in `.venv/`.
- Install dependencies commonly used for SAC:
  - `numpy`, `torch`, `gymnasium`, `tensorboard` (adjust as the code requires).
- Run training: `python train_sac.py`.
- Monitor: `tensorboard --logdir tb`.

## Code Conventions
- Language: Python 3.9+ recommended.
- Style: PEP8; prefer clear, small functions; avoid unnecessary abstractions.
- Types: Add type hints where they clarify intent; don’t overdo it.
- Naming: `snake_case` for functions and variables; `CamelCase` for classes; constants in `UPPER_SNAKE_CASE`.
- Logging: Prefer lightweight logging over noisy prints. Direct metrics to TensorBoard.

## Repository Organization
- Environments: place custom envs as `env_*.py` at the root (or in `envs/` if the project grows).
- Trainers/Scripts: place experiment entrypoints as `train_*.py`.
- Artifacts: logs and checkpoints should not be committed; see `.gitignore`.
- Tests (if added): place under `tests/` and keep them minimal and fast.

## Agent Guidance
- Make minimal, surgical changes; avoid refactors unless explicitly requested.
- Do not commit large artifacts (model weights, logs). Keep outputs in `tb/`, `runs/`, or `checkpoints/` (all ignored by Git).
- If adding dependencies, prefer widely-used, actively maintained packages. Avoid heavyweight additions without need.
- If ambiguity exists (e.g., expected CLI args), default to sensible behavior and document it in `README.md`.
- Respect existing file names and interfaces (e.g., don’t rename `env_energyhub.py` or `train_sac.py` without instruction).

## Reproducibility
- Consider setting random seeds (NumPy, PyTorch, environment) in training scripts.
- Document non-default hyperparameters in `README.md` if you change them.

## Safety & Secrets
- Never commit secrets. Use a local `.env` and ensure it’s ignored.
- Validate that newly added logging does not leak sensitive values.

## When Expanding
- If the project grows, consider moving envs to `envs/` and trainers to `trainers/` or `experiments/` while maintaining clear module boundaries.
- Introduce configs (YAML/TOML) only when needed; keep defaults simple.

