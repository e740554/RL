import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EnergyHubEnv(gym.Env):
    """
    POC Config 1: Electrolyzer + MD + Heat Exchanger
    State:  [x_h2, x_upw, p_ren] in [0,1]
    Action: [u_ely, u_split, u_md] in [0,1]
    Goal: heat-matching without starving UPW & keep H2 throughput high
    """
    metadata = {"render_modes": []}

    def __init__(self, seed=None, max_steps=240):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        
        # --- time step ---
        self.dt_h = np.float32(1/60)      # 1 minute    
        
        # --- process constants (POC scale) ---
        self.P_ELY_MAX      = np.float32(100.0)   # kW
        self.kWh_per_kg_H2  = np.float32(50.0)    # kWh / kg H2
        self.ETA_WASTE      = np.float32(0.20)    # fraction of P_ely into low-grade heat
        self.Q_MD_MAX       = np.float32(30.0)    # kW_th demand at u_md = 1
        self.kWhth_per_m3   = np.float32(100.0)   # kWh_th per m3 UPW
        self.WATER_PER_KG   = np.float32(0.010)   # m3 water per kg H2

        self.UPW_CAP = np.float32(5.0)            # m3
        self.H2_CAP  = np.float32(100.0)          # kg

        # reward weights--- Normalized  
        # tip: a1 high enough so H2 matters; a3 moderate; BIG is hard penalty
        self.a1  = np.float32(10.0)
        self.a3  = np.float32(0.5)
        self.BIG = np.float32(5.0)

        # ---- spaces ----
        # obs: [x_h2, x_upw, p_ren] in [0,1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        # act: [u_ely, u_split, u_md] in [0,1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.max_steps = int(max_steps)
        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        # start with half-full buffers; randomize renewables
        self.x_h2  = np.float32(0.5)
        self.x_upw = np.float32(0.5)
        # renewable availability varies per episode
        self.p_ren = float(self.rng.uniform(0.6, 1.0))
        self.step_idx = 0        
        obs = np.array([self.x_h2, self.x_upw, self.p_ren], dtype=np.float32)
        return obs, {}

    def step(self, action):
        u_ely, u_split, u_md = np.clip(action, 0.0, 1.0)

        # soft safety: throttle if UPW is nearly empty
        throttle = 1.0 / (1.0 + np.exp(-40*(self.x_upw-0.1)))
        u_ely *= throttle
        
        # --- electrolyzer ---
        P_ely = u_ely * self.P_ELY_MAX * self.p_ren             # kW
        h2_prod = (P_ely * self.dt_h) / self.kWh_per_kg_H2      # kg/step
        Q_waste = self.ETA_WASTE * P_ely                        # kW_th
        
        # --- membrane distillation (MD) ---
        Q_md_need = u_md * self.Q_MD_MAX                        # kW_th
        Q_avail = u_split * Q_waste                             # kW_th    
        Q_used = min(Q_avail, Q_md_need)                        # kW_th    
        upw_make = (Q_used * self.dt_h) / self.kWhth_per_m3     # m3/step

        # --- UPW consumption by electrolyzer ---
        upw_use  = h2_prod * self.WATER_PER_KG                  # m3/step

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
        heat_unutilized = np.maximum(0.0, Q_avail - Q_used)   # waste heat not used by MD
        md_shortage     = np.maximum(0.0, Q_md_need - Q_used) # MD wanted more heat than it got
        denom = (self.Q_MD_MAX + 1e-9)
        norm_unused = heat_unutilized / denom
        norm_short  = md_shortage     / denom

        # reward with normalized penalties
        reward = (
            self.a1 * float(h2_prod)
            - self.a3 * float(norm_unused + norm_short)
            - self.BIG * float(violation)
        )

        self.step_idx += 1
        terminated = False
        truncated = self.step_idx >= self.max_steps

        obs = np.array([self.x_h2, self.x_upw, self.p_ren], dtype=np.float32)

        info = {
            "h2_prod": float(h2_prod),
            "upw_make": float(upw_make),
            "norm_unused": float(norm_unused),
            "norm_short": float(norm_short),
            "violations": float(violation),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
