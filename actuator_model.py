import numpy as np

# ---- From hardware ----
V_SUPPLY = 12.0          # [V]
FREQ_HZ = 4.5            # [Hz] firmware drive frequency
STROKE = 19.7e-3         # [m] total stroke
X_MAX = STROKE / 2       # [m] travel each side of center

P_NOMINAL = dict(
    m=0.1076,     # moving mass [kg]: coil + 2 screws + plate (SolidWorks mass properties)
    m_a=0.186,    # added mass [kg]: (8/3)*rho*r^3, equivalent radius r = sqrt(A/pi) = 41.2 mm (ESTIMATE)
    Kf=5.0,       # force constant [N/A] (ESTIMATE, typical 12 V VCA)
    Kb=5.0,       # back-EMF constant [V*s/m] (equals Kf in SI units)
    R=4.0,        # coil resistance [ohm] (ESTIMATE)
    k=0,     # silicone membrane stiffness [N/m] (free)
    b=1.0,        # mechanical damping [N*s/m] (ESTIMATE)
    c_d=2.9932,   # quadratic drag [N*s^2/m^2] = 0.5*1000*0.005345*1.12 (flat plate, Cd = 1.12)
    k_stop=1e5,   # end-stop stiffness [N/m]
    b_stop=50.0,  # end-stop damping [N*s/m]
)

def firmware_voltage(t):
    """Replicates the ESP32 firmware: full +V (push) / -V (pull) square wave."""
    half_period = 1.0 / (2 * FREQ_HZ)
    return V_SUPPLY if int(t // half_period) % 2 == 0 else -V_SUPPLY

def accel(x, v, V, p):
    """Acceleration and current. Inductance neglected: L/R = 0.25 ms << 1 ms control step."""
    i = (V - p["Kb"] * v) / p["R"]
    F_fluid = p["c_d"] * abs(v) * v
    if abs(x) > X_MAX:
        F_stop = -(p["k_stop"] * (x - np.sign(x) * X_MAX) + p["b_stop"] * v)
    else:
        F_stop = 0.0
    a = (p["Kf"]*i - p["b"]*v - p["k"]*x - F_fluid + F_stop) / (p["m"] + p["m_a"])
    return a, i

def step(x, v, V, p, dt=1e-3, substeps=10):
    """Advance the actuator by one control step with voltage V held constant (RK4)."""
    h = dt / substeps
    f = lambda x, v: (v, accel(x, v, V, p)[0])
    for _ in range(substeps):
        k1x, k1v = f(x, v)
        k2x, k2v = f(x + 0.5*h*k1x, v + 0.5*h*k1v)
        k3x, k3v = f(x + 0.5*h*k2x, v + 0.5*h*k2v)
        k4x, k4v = f(x + h*k3x, v + h*k3v)
        x += h/6 * (k1x + 2*k2x + 2*k3x + k4x)
        v += h/6 * (k1v + 2*k2v + 2*k3v + k4v)
    return x, v, accel(x, v, V, p)[1]