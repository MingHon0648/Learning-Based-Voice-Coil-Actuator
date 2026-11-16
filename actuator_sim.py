import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from actuator_model import P_NOMINAL, V_SUPPLY, FREQ_HZ, STROKE, X_MAX, firmware_voltage

# Full model = the shared nominal parameters plus coil inductance
p = dict(P_NOMINAL, L=1e-3)   # coil inductance [H] (estimate)

def dynamics(t, s, p, voltage_fn):
    x, v, i = s                                   # position, velocity, current (state vector)
    V = voltage_fn(t)
    F_fluid = p["c_d"] * abs(v) * v               # quadratic water drag
    if abs(x) > X_MAX:                            # End stops: a stiff spring + damper that only engages beyond the travel limit
        F_stop = -(p["k_stop"] * (x - np.sign(x) * X_MAX) + p["b_stop"] * v)
    else:
        F_stop = 0.0
    dv = (p["Kf"]*i - p["b"]*v - p["k"]*x - F_fluid + F_stop) / (p["m"] + p["m_a"])
    di = (V - p["R"]*i - p["Kb"]*v) / p["L"]            # Kirchhoff's voltage law
    return [v, dv, di]

T_END = 1.0  # simulate 1 s
sol = solve_ivp(dynamics, (0, T_END), [0, 0, 0], args=(p, firmware_voltage),
                method="LSODA", max_step=1e-4)

t = sol.t
x, v, i = sol.y
V = np.array([firmware_voltage(tk) for tk in t])

# Measure steady-state stroke (ignore the first 0.5 s start-up)
steady = t > 0.5
print(f"Peak-to-peak stroke: {(x[steady].max() - x[steady].min())*1000:.1f} mm "
      f"(max possible {STROKE*1000:.1f} mm)")
print(f"Peak current: {np.abs(i).max():.2f} A")

fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
ax[0].plot(t, V); ax[0].set_ylabel("Voltage [V]")
ax[1].plot(t, x*1000)
ax[1].axhline(X_MAX*1000, ls="--", c="gray"); ax[1].axhline(-X_MAX*1000, ls="--", c="gray")
ax[1].set_ylabel("Position [mm]")
ax[2].plot(t, i); ax[2].set_ylabel("Current [A]")
ax[2].set_xlabel("Time [s]")
ax[0].set_title("Open-loop firmware drive (simulated)")
plt.tight_layout(); plt.savefig("figures/open_loop_sim.png", dpi=150); plt.show()