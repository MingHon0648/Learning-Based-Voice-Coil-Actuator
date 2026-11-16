import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from voice_coil_env import VoiceCoilEnv

model = PPO.load("ppo_nominal")
env = VoiceCoilEnv()

def run(policy, seed=1):
    obs, _ = env.reset(seed=seed)
    log = {"x": [], "ref": [], "V": []}
    for _ in range(env.max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        log["x"].append(info["x"]); log["ref"].append(info["ref"]); log["V"].append(action[0] * 12)
    return {k: np.array(v) for k, v in log.items()}

policies = {
    "PPO": lambda o: model.predict(o, deterministic=True)[0],
    "P-controller": lambda o: np.array([np.clip(4.0 * o[4], -1, 1)], dtype=np.float32),
}

t = np.arange(1, env.max_steps + 1) * env.dt
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for name, pol in policies.items():
    r = run(pol)
    rms = np.sqrt(np.mean((r["x"][500:] - r["ref"][500:])**2)) * 1000
    print(f"{name:13s} RMS error {rms:.2f} mm | peak voltage {np.abs(r['V']).max():.1f} V")
    ax[0].plot(t, r["x"]*1000, label=name)
    ax[1].plot(t, r["V"], label=name)
ax[0].plot(t, r["ref"]*1000, "k--", label="Reference")
ax[0].set_ylabel("Position [mm]"); ax[1].set_ylabel("Voltage [V]"); ax[1].set_xlabel("Time [s]")
ax[0].legend(); ax[0].set_title("PPO vs. P-controller (nominal parameters)")
plt.tight_layout(); plt.show()