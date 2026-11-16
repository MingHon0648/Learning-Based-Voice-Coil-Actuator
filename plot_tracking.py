"""Time-series comparison of all four controllers on the nominal actuator.

Saves figures/tracking.png. Uses the same controllers as evaluate_all.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from actuator_model import V_SUPPLY, X_MAX
from voice_coil_env import VoiceCoilEnv
from evaluate_all import OpenLoop, PID, SB3Policy   # same controllers as the evaluation

SEED = 3
COLORS = {"Open-loop firmware": "tab:gray", "PID": "tab:orange",
          "PPO (nominal)": "tab:blue", "PPO (DR)": "tab:green"}
controllers = {
    "Open-loop firmware": OpenLoop(),
    "PID": PID(),
    "PPO (nominal)": SB3Policy("ppo_nominal"),
    "PPO (DR)": SB3Policy("ppo_dr"),
}

def run(ctrl):
    env = VoiceCoilEnv()
    obs, _ = env.reset(seed=SEED)
    ctrl.reset()
    log = {"x": [], "ref": [], "V": []}
    for _ in range(env.max_steps):
        a = ctrl(obs)
        obs, _, _, _, info = env.step(a)
        log["x"].append(info["x"] * 1000)
        log["ref"].append(info["ref"] * 1000)
        log["V"].append(float(np.clip(a[0], -1, 1)) * V_SUPPLY)
    return {k: np.array(v) for k, v in log.items()}, env.dt

fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1.2, 1.2]})
for name, ctrl in controllers.items():
    r, dt = run(ctrl)
    t = np.arange(1, len(r["x"]) + 1) * dt
    c = COLORS[name]
    ax[0].plot(t, r["x"], label=name, lw=1.4, c=c, alpha=0.6 if name.startswith("Open") else 1)
    if name != "Open-loop firmware":
        ax[1].plot(t, r["x"] - r["ref"], label=name, lw=1.0, c=c)
    ax[2].plot(t, r["V"], label=name, lw=0.8, c=c, alpha=0.6 if name.startswith("Open") else 0.9)
ax[0].plot(t, r["ref"], "k--", lw=1.2, label="Reference")
for y in (X_MAX * 1000, -X_MAX * 1000):
    ax[0].axhline(y, ls=":", c="k", lw=0.8)
ax[0].text(0.005, X_MAX * 1000 - 0.3, "end stop", va="top", fontsize=8)
ax[0].set_ylabel("Position [mm]")
ax[1].set_ylabel("Tracking error [mm]")
ax[2].set_ylabel("Voltage [V]"); ax[2].set_xlabel("Time [s]")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=8, frameon=False)
ax[1].set_ylim(-1.5, 1.5)   # start-up transient clipped; steady-state error is the interest
ax[0].set_ylim(-12.5, 12.5)
ax[0].set_title("4.5 Hz, ±7 mm sine tracking on the nominal actuator")
plt.tight_layout(); plt.savefig("figures/tracking.png", dpi=150); plt.show()
