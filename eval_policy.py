# eval_policy.py
import numpy as np
from stable_baselines3 import SAC
from env_energyhub import EnergyHubEnv


def eval_once(model, max_steps=720, seed=123):
    env = EnergyHubEnv(seed=seed, max_steps=max_steps)
    obs, _ = env.reset(seed=seed)

    totals = dict(
        reward=0.0,
        h2=0.0,
        upw=0.0,
        norm_unused=0.0,
        norm_short=0.0,
        viol=0.0,
        steps=0,
    )

    done = False
    trunc = False
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        totals["reward"] += r
        totals["h2"] += info["h2_prod"]
        totals["upw"] += info["upw_make"]
        totals["norm_unused"] += info["norm_unused"]
        totals["norm_short"] += info["norm_short"]
        totals["viol"] += info["violations"]
        totals["steps"] += 1

    # episode-level KPIs
    steps = max(1, totals["steps"])
    kpis = {
        "ep_reward": totals["reward"],
        "H2_kg": totals["h2"],
        "UPW_m3": totals["upw"],
        "avg_norm_unused": totals["norm_unused"] / steps,
        "avg_norm_short": totals["norm_short"] / steps,
        "violations": totals["viol"],
        "steps": steps,
    }
    return kpis


if __name__ == "__main__":
    model = SAC.load("sac_energyhub_cpu", device="cpu")
    ks = [eval_once(model, max_steps=720, seed=100 + i) for i in range(5)]
    # summarize
    def meanstd(x):
        a = np.array(x)
        return float(a.mean()), float(a.std())

    print("\nDeterministic eval over 5 episodes:")
    for k in ks[0].keys():
        vals = [r[k] for r in ks]
        m, s = meanstd(vals)
        print(f"{k:>18s}: {m:.3f} +/- {s:.3f}")

