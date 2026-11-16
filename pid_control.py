import numpy as np
import matplotlib.pyplot as plt
from actuator_model import P_NOMINAL, V_SUPPLY, X_MAX, FREQ_HZ, step, firmware_voltage

DT = 1e-3          # control period [s] (1 kHz loop)
T_END = 2.0        # [s]
NOISE = 0.05e-3    # position sensor noise, standard deviation [m]
AMP = 7e-3         # reference amplitude [m], inside the ±9.85 mm stroke
KP, KI, KD = 5000.0, 50000.0, 50.0
TAU_D = 0.002      # derivative filter time constant [s]

rng = np.random.default_rng(0)

def reference(t):
    return AMP * np.sin(2 * np.pi * FREQ_HZ * t)

def open_loop(t, x_meas, state):
    return firmware_voltage(t)          # ignores the sensor, like the ESP32 firmware

def pid(t, x_meas, state, tau_d=TAU_D):
    err = reference(t) - x_meas
    raw_deriv = (err - state["prev_err"]) / DT
    state["prev_err"] = err
    alpha = DT / (tau_d + DT)                                # first-order low-pass filter
    state["deriv"] += alpha * (raw_deriv - state["deriv"])
    V_cmd = KP*err + KI*state["integral"] + KD*state["deriv"]
    V = float(np.clip(V_cmd, -V_SUPPLY, V_SUPPLY))   # driver can't exceed supply voltage
    if V == V_cmd:                                    # anti-windup
        state["integral"] += err * DT
    return V

def pid_no_filter(t, x_meas, state):
    return pid(t, x_meas, state, tau_d=0.0)           # ablation: raw finite-difference derivative

def run(controller, p=P_NOMINAL):
    x = v = 0.0
    state = {"integral": 0.0, "prev_err": 0.0, "deriv": 0.0}
    log = {k: [] for k in ("t", "x", "ref", "V", "i")}
    for k in range(int(T_END / DT)):
        t = k * DT
        x_meas = x + rng.normal(0, NOISE)             # noisy position sensor
        V = controller(t, x_meas, state)
        x, v, i = step(x, v, V, p, DT)
        for key, val in zip(log, (t + DT, x, reference(t + DT), V, i)):
            log[key].append(val)
    return {k: np.array(a) for k, a in log.items()}

def report(name, r):
    s = r["t"] > 1.0                                  # ignore start-up
    rms = np.sqrt(np.mean((r["x"][s] - r["ref"][s])**2)) * 1000
    stops = np.mean(np.abs(r["x"][s]) > X_MAX) * 100
    chatter = np.mean(np.abs(np.diff(r["V"][s])))      # mean |dV| between control steps
    print(f"{name:16s} RMS error {rms:5.2f} mm | time at end stops {stops:4.1f}% "
          f"| chatter {chatter:4.2f} V/step | peak current {np.abs(r['i']).max():.2f} A")

results = {"Open-loop": run(open_loop), "PID": run(pid)}
for name, r in results.items():
    report(name, r)
report("PID (no filter)", run(pid_no_filter))        # derivative-filter ablation (not plotted)

fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
ref = results["PID"]
ax[0].plot(ref["t"], ref["ref"]*1000, "k--", label="Reference")
for name, r in results.items():
    ax[0].plot(r["t"], r["x"]*1000, label=name)
    ax[1].plot(r["t"], r["V"], label=name)
    ax[2].plot(r["t"], r["i"], label=name)
for a in ax[0:1]:
    a.axhline(X_MAX*1000, ls=":", c="gray"); a.axhline(-X_MAX*1000, ls=":", c="gray")
ax[0].set_ylabel("Position [mm]"); ax[1].set_ylabel("Voltage [V]")
ax[2].set_ylabel("Current [A]"); ax[2].set_xlabel("Time [s]")
ax[0].legend(loc="upper right"); ax[0].set_title("Open-loop firmware vs. PID tracking")
plt.tight_layout(); plt.savefig("figures/pid_vs_open_loop.png", dpi=150); plt.show()