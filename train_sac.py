# train_sac.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)  # avoid torch oversubscribing CPU threads

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import numpy as np

from env_energyhub import EnergyHubEnv

def make_env(rank, max_steps=240, seed=42):
    def _thunk():
        env = EnergyHubEnv(seed=seed+rank, max_steps=max_steps)
        return Monitor(env)
    return _thunk

class InfoLogger(BaseCallback):
    """Log mean of selected info keys to TensorBoard."""
    def __init__(self, keys, verbose=0):
        super().__init__(verbose); self.keys = keys
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos and isinstance(infos[0], dict):
            for k in self.keys:
                vals = [i.get(k) for i in infos if k in i]
                if vals:
                    self.logger.record_mean(f"env/{k}", float(np.mean(vals)))
        return True

if __name__ == "__main__":
    num_envs = 6  # try 4–8 depending on cores
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecMonitor(env)

    eval_env = EnergyHubEnv(seed=999, max_steps=240)
    eval_cb  = EvalCallback(eval_env, eval_freq=5000, n_eval_episodes=3, deterministic=True)
    info_cb  = InfoLogger(keys=["h2_prod","upw_make","norm_unused","norm_short","violations"])
    callbacks = CallbackList([eval_cb, info_cb])

    # logger
    new_logger = configure("./tb", ["stdout", "tensorboard"])

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tb",
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=128,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=2000,
        policy_kwargs=dict(net_arch=[64, 64]),
        device="cpu"
    )
    model.set_logger(new_logger)
    model.learn(total_timesteps=200_000, callback=callbacks)
    model.save("sac_energyhub_cpu")
