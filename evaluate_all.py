import csv
import numpy as np
from stable_baselines3 import PPO
from actuator_model import P_NOMINAL, V_SUPPLY, X_MAX, FREQ_HZ
from voice_coil_env import VoiceCoilEnv

DT = 1e-3
SEEDS = range(10)   # 10 episodes per test, each starting at a different point in the cycle

# ---------------- Controllers ----------------
class OpenLoop:
    """Replica of the ESP32 firmware: +/-12 V square wave, ignores the sensor."""
    def reset(self): self.k = 0
    def __call__(self, obs):
        t = self.k * DT; self.k += 1
        half_period = 1 / (2 * FREQ_HZ)
        return np.array([1.0 if int(t // half_period) % 2 == 0 else -1.0], dtype=np.float32)

class PID:
    """Same gains and derivative filter as pid_control.py."""
    KP, KI, KD, TAU_D = 5000.0, 50000.0, 50.0, 0.002
    def reset(self):
        self.integral = self.prev_err = self.deriv = 0.0
        self.first = True
    def __call__(self, obs):
        err = (obs[2] - obs[0]) * X_MAX              # reference minus measured position [m]
        if self.first: self.prev_err, self.first = err, False
        raw = (err - self.prev_err) / DT
        self.prev_err = err
        self.deriv += DT / (self.TAU_D + DT) * (raw - self.deriv)
        V_cmd = self.KP*err + self.KI*self.integral + self.KD*self.deriv
        V = float(np.clip(V_cmd, -V_SUPPLY, V_SUPPLY))
        if V == V_cmd: self.integral += err * DT     # anti-windup
        return np.array([V / V_SUPPLY], dtype=np.float32)

class SB3Policy:
    def __init__(self, path): self.model = PPO.load(path)
    def reset(self): pass
    def __call__(self, obs): return self.model.predict(obs, deterministic=True)[0]

controllers = {
    "Open-loop": OpenLoop(),
    "PID": PID(),
    "PPO (nominal)": SB3Policy("ppo_nominal"),
    "PPO (DR)": SB3Policy("ppo_dr"),
}

# ---------------- Test conditions ----------------
def scaled(**factors):
    p = dict(P_NOMINAL)
    for k, f in factors.items():
        p[k] = P_NOMINAL[k] * f
    p["Kb"] = p["Kf"]
    return p

CONDITIONS = {
    # Inside the domain-randomization training ranges
    "Nominal":               scaled(),
    "Heavy drag (3x)":       scaled(c_d=3.0),
    "Hot coil (R 1.25x)":    scaled(R=1.25),
    # OUTSIDE the training ranges (never seen by PPO-DR)
    "Drag 6x":               scaled(c_d=6.0),
    "Added mass 2.5x":       scaled(m_a=2.5),
    "Friction 6x":           scaled(b=6.0),
    "Weak magnet (Kf 0.6x)": scaled(Kf=0.6),
    "Combined worst case":   scaled(c_d=6.0, m_a=2.0, b=4.0, Kf=0.7, R=1.4),
}

def evaluate(ctrl, params):
    rms, chatter, stops = [], [], []
    for seed in SEEDS:
        env = VoiceCoilEnv(params=params)
        obs, _ = env.reset(seed=seed)
        ctrl.reset()
        xs, refs, Vs = [], [], []
        for _ in range(env.max_steps):
            a = ctrl(obs)
            obs, _, _, _, info = env.step(a)
            xs.append(info["x"]); refs.append(info["ref"])
            Vs.append(float(np.clip(a[0], -1, 1)) * V_SUPPLY)
        xs, refs, Vs = map(np.array, (xs, refs, Vs))
        s = slice(500, None)                          # skip start-up
        rms.append(np.sqrt(np.mean((xs[s] - refs[s])**2)) * 1000)
        chatter.append(np.mean(np.abs(np.diff(Vs[s]))))
        stops.append(np.mean(np.abs(xs[s]) > X_MAX) * 100)
    return np.mean(rms), np.mean(chatter), np.mean(stops)

if __name__ == "__main__":
    rows = []
    print(f"{'Condition':24s} {'Controller':14s} {'RMS err [mm]':>12s} {'Chatter [V/step]':>17s} {'At stops [%]':>13s}")
    for cname, params in CONDITIONS.items():
        for name, ctrl in controllers.items():
            r, c, s = evaluate(ctrl, params)
            rows.append((cname, name, r, c, s))
            print(f"{cname:24s} {name:14s} {r:12.2f} {c:17.2f} {s:13.1f}")
        print()

    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "controller", "rms_mm", "chatter_V_per_step", "stops_pct"])
        w.writerows(rows)
    print("Saved results.csv")