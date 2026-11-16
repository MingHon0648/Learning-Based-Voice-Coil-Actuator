import numpy as np
from gymnasium.utils.env_checker import check_env
from voice_coil_env import VoiceCoilEnv

env = VoiceCoilEnv()
check_env(env)                      # Gymnasium's official sanity checks
print("Environment passed check_env")

from actuator_model import X_MAX

def evaluate(name, policy, seeds=(1,)):
    totals, terms, rmss = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)                     # makes the random baseline reproducible
        total, errors = 0.0, []
        track = change = stop = 0.0; prev = 0.0
        for _ in range(env.max_steps):
            a = policy(obs); a_c = float(np.clip(a[0], -1, 1))
            obs, reward, terminated, truncated, info = env.step(a)
            total += reward
            errors.append(info["x"] - info["ref"])
            # the three reward terms, logged separately
            track += ((info["x"] - info["ref"]) / X_MAX)**2
            change += 0.01 * (a_c - prev)**2; prev = a_c
            stop += float(abs(info["x"]) > X_MAX)
        totals.append(total); terms.append((track, change, stop))
        rmss.append(np.sqrt(np.mean(np.square(errors[500:]))) * 1000)
    tr, ch, st = np.mean(terms, axis=0)
    spread = f" +/- {np.std(totals):5.1f}" if len(seeds) > 1 else " " * 9
    print(f"{name:22s} total reward {np.mean(totals):7.1f}{spread} | tracking -{tr:6.1f} "
          f"| action change -{ch:5.2f} | end stop -{st:5.1f} | RMS error {np.mean(rmss):.2f} mm")

evaluate("Zero voltage", lambda o: np.array([0.0], dtype=np.float32))
evaluate("Random actions (30 ep)", lambda o: env.action_space.sample(), seeds=range(30))
evaluate("Simple P-controller", lambda o: np.array([np.clip(4.0 * o[4], -1, 1)], dtype=np.float32))