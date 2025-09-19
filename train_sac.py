import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env_energyhub import EnergyHubEnv

def make_env():
    return Monitor(EnergyHubEnv())

if __name__ == "__main__":
    env = DummyVecEnv([make_env])  # SB3 prefers VecEnvs
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tb",
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=200_000,
        batch_size=256,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        policy_kwargs=dict(net_arch=[128, 128])
    )
    model.learn(total_timesteps=150_000)
    model.save("sac_energyhub")
