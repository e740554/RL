import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EnergyHubEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=None, max_steps: int = 720):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        # ---- constants ----
        self.dt_h = 1/60
        self.P_ELY_MAX = 100.0        # kW
        self.kWh_per_kg_H2 = 50.0     # kWh/kg
        self.ETA_WASTE = 0.20         # fraction of P_ely as low-grade heat
        self.Q_MD_MAX = 30.0          # kW_th (at u_md=1)
        self.kWhth_per_m3 = 100.0     # kWh_th per m3 of UPW
        self.WATER_PER_KG = 0.010     # m3/kg H2

        self.UPW_CAP = 5.0            # m3
        self.H2_CAP  = 100.0          # kg

        # reward weights
        self.a1, self.a3, self.BIG = 10.0, 0.5, 5.0

        # ---- spaces ----
        # obs: [x_h2, x_upw, p_ren] in [0,1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        # act: [u_ely, u_split, u_md] in [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        # episode length
        self.max_steps = int(max_steps)
        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        # start with half-full buffers; randomize renewables
        self.x_h2  = 0.5
        self.x_upw = 0.5
        self.step_idx = 0
        self.p_ren = float(self.rng.uniform(0.6, 1.0))
        obs = np.array([self.x_h2, self.x_upw, self.p_ren], dtype=np.float32)
        return obs, {}

    def step(self, action):
        u_ely, u_split, u_md = np.clip(action, 0.0, 1.0)

        # soft safety: throttle if UPW is nearly empty
        throttle = 1.0 / (1.0 + np.exp(-40*(self.x_upw-0.1)))
        u_ely *= throttle

        P_ely = u_ely * self.P_ELY_MAX * self.p_ren
        h2_prod = (P_ely * self.dt_h) / self.kWh_per_kg_H2      # kg
        Q_waste = self.ETA_WASTE * P_ely                        # kW_th
        Q_md_need = u_md * self.Q_MD_MAX
        Q_avail = u_split * Q_waste
        Q_used = min(Q_avail, Q_md_need)

        upw_make = (Q_used * self.dt_h) / self.kWhth_per_m3     # m3
        upw_use  = h2_prod * self.WATER_PER_KG                  # m3

        # buffer updates (normalized)
        upw_next = self.x_upw * self.UPW_CAP + upw_make - upw_use
        h2_next  = self.x_h2  * self.H2_CAP  + h2_prod

        violation = 0.0
        if upw_next < 0.0:
            violation += 1.0
            upw_next = 0.0
        if h2_next > self.H2_CAP:
            violation += 1.0
            h2_next = self.H2_CAP

        self.x_upw = np.clip(upw_next / self.UPW_CAP, 0.0, 1.0)
        self.x_h2  = np.clip(h2_next  / self.H2_CAP,  0.0, 1.0)

        # normalized, interpretable heat terms
        heat_unutilized = max(0.0, Q_avail - Q_used)   # waste heat not used by MD
        md_shortage     = max(0.0, Q_md_need - Q_used) # MD wanted more heat than it got
        denom = (self.Q_MD_MAX + 1e-9)
        norm_unused = heat_unutilized / denom
        norm_short  = md_shortage     / denom

        # reward with normalized penalties
        reward = (
            self.a1 * h2_prod
            - self.a3 * (norm_unused + norm_short)
            - self.BIG * violation
        )

        self.step_idx += 1
        terminated = False
        truncated = self.step_idx >= self.max_steps
        obs = np.array([self.x_h2, self.x_upw, self.p_ren], dtype=np.float32)
        # keep heat_mismatch for continuity in logs, add new diagnostics
        heat_mismatch = abs(Q_md_need - Q_avail)
        info = {
            "h2_prod": h2_prod,
            "upw_make": upw_make,
            "heat_mismatch": heat_mismatch,
            "heat_unutilized": heat_unutilized,
            "md_shortage": md_shortage,
            "norm_unused": norm_unused,
            "norm_short": norm_short,
            "violations": violation
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
