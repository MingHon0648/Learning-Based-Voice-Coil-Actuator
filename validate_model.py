import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from actuator_model import P_NOMINAL, X_MAX, firmware_voltage, step

p = dict(P_NOMINAL, L=1e-3)   # same parameters, plus inductance for the full model

def full_model(t, s):
    """Full model with coil inductance (same equations as actuator_sim.py)."""
    x, v, i = s
    V = firmware_voltage(t)
    F_stop = -(p["k_stop"]*(x - np.sign(x)*X_MAX) + p["b_stop"]*v) if abs(x) > X_MAX else 0.0
    dv = (p["Kf"]*i - p["b"]*v - p["k"]*x - p["c_d"]*abs(v)*v + F_stop) / (p["m"] + p["m_a"])
    di = (V - p["R"]*i - p["Kb"]*v) / p["L"]
    return [v, dv, di]

def full_model_V(t, s, V):
    """Full model with the voltage held constant (one control step)."""
    x, v, i = s
    F_stop = -(p["k_stop"]*(x - np.sign(x)*X_MAX) + p["b_stop"]*v) if abs(x) > X_MAX else 0.0
    dv = (p["Kf"]*i - p["b"]*v - p["k"]*x - p["c_d"]*abs(v)*v + F_stop) / (p["m"] + p["m_a"])
    di = (V - p["R"]*i - p["Kb"]*v) / p["L"]
    return [v, dv, di]

DT = 1e-3
t = np.arange(1, 1001) * DT
full = solve_ivp(full_model, (0, 1), [0, 0, 0], method="LSODA", max_step=1e-4, t_eval=t)

x = v = 0.0
x_simple = []
for k in range(1000):
    x, v, _ = step(x, v, firmware_voltage(k * DT), P_NOMINAL, DT)
    x_simple.append(x)
x_simple = np.array(x_simple)

diff = np.abs(x_simple - full.y[0]) * 1000
print(f"RMS difference: {np.sqrt(np.mean(diff**2)):.2f} mm | max difference: {diff.max():.2f} mm")

# ---- Cost per 1 ms control step, as used inside the RL environment ----
# The voltage changes every step (zero-order hold), so an adaptive solver must be restarted each step.
t0 = time.perf_counter(); s = [0.0, 0.0, 0.0]
for k in range(1000):
    V_k = firmware_voltage(k * DT)
    sol = solve_ivp(lambda t, y: full_model_V(t, y, V_k), (k*DT, (k+1)*DT), s, method="LSODA", max_step=1e-4)
    s = sol.y[:, -1]
t_lsoda = (time.perf_counter() - t0) / 1000
t0 = time.perf_counter(); x = v = 0.0
for k in range(1000):
    x, v, _ = step(x, v, firmware_voltage(k * DT), P_NOMINAL, DT)
t_rk4 = (time.perf_counter() - t0) / 1000
print(f"Per control step: full model + LSODA {t_lsoda*1e6:.0f} us | reduced model + RK4 {t_rk4*1e6:.0f} us "
      f"({t_lsoda/t_rk4:.1f}x faster)")

plt.plot(t, full.y[0]*1000, label="Full model (solve_ivp, with inductance)")
plt.plot(t, x_simple*1000, "--", label="Simplified model (RK4, no inductance)")
plt.xlabel("Time [s]"); plt.ylabel("Position [mm]"); plt.legend()
plt.title("Model simplification check"); plt.tight_layout(); plt.show()