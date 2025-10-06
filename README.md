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

## Customizing Rewards

The reward structure is defined in `env_energyhub.py` (lines 32-36 and 105-110). You can modify the reward weights to change the training behavior:

```python
# Reward weights in EnergyHubEnv.__init__()
self.a1  = np.float32(10.0)  # H2 production reward weight
self.a3  = np.float32(0.5)   # Heat inefficiency penalty weight
self.BIG = np.float32(5.0)   # Constraint violation penalty weight

# Reward calculation in step()
reward = (
    self.a1 * float(h2_prod)                        # reward H2 production
    - self.a3 * float(norm_unused + norm_short)     # penalize heat inefficiency
    - self.BIG * float(violation)                   # penalize constraint violations
)
```

**Tuning tips:**
- Increase `a1` to prioritize H2 production
- Increase `a3` to emphasize heat matching efficiency
- Increase `BIG` to more strongly penalize constraint violations
- After changing rewards, retrain the model to see the effect

## Evaluating Trained Policies

Use `eval_policy.py` to evaluate a trained model:

```bash
# Evaluate the saved model over 5 episodes
python eval_policy.py
```

The script will output episode-level metrics including:
- Episode reward
- H2 production (kg)
- UPW production (m³)
- Average heat inefficiencies
- Constraint violations

**Customizing evaluation:**
- Change the number of episodes: modify `range(5)` in line 50
- Change episode length: modify `max_steps=720` parameter
- Change model path: modify `"sac_energyhub_cpu"` in line 49

## Running Additional Training

To implement another training run with different parameters:

1. **Modify hyperparameters** in `train_sac.py`:
   ```python
   # Training parameters (lines 51-68)
   learning_rate=3e-4,     # try 1e-4 or 1e-3
   batch_size=128,         # try 64 or 256
   total_timesteps=200_000 # try 500_000 for longer training
   ```

2. **Change model save name** to avoid overwriting:
   ```python
   model.save("sac_energyhub_v2")  # line 69
   ```

3. **Adjust environment parameters** if needed:
   ```python
   # In make_env() function (line 19)
   max_steps=240  # episode length

   # In EnergyHubEnv.__init__()
   # Modify reward weights or process constants
   ```

4. **Run training**:
   ```bash
   python train_sac.py
   ```

5. **Monitor with TensorBoard**:
   ```bash
   tensorboard --logdir tb
   ```

**Example workflow for experimentation:**
1. Modify reward weights in `env_energyhub.py`
2. Update model save name in `train_sac.py`
3. Run training: `python train_sac.py`
4. Evaluate: update model path in `eval_policy.py` and run
5. Compare results using TensorBoard logs

## Notes
- GPU is optional; if available and supported by your PyTorch install, training will typically be faster.
- For reproducibility, consider fixing seeds (NumPy, PyTorch, environment) in your training script.
- Keep large artifacts (models, logs) out of Git; use the `tb/`, `checkpoints/`, or external storage.
- Each training run creates logs in `tb/` with timestamps - useful for comparing different experiments.

## License
No license has been specified. Add a license if you plan to share or distribute this code.

