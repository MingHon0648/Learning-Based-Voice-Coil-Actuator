import numpy as np
import gymnasium as gym
from gymnasium import spaces
from actuator_model import P_NOMINAL, V_SUPPLY, X_MAX, FREQ_HZ, step

class VoiceCoilEnv(gym.Env):
    """Track a 4.5 Hz sine reference with the fluid-loaded voice coil actuator."""

    def __init__(self, params=None, dt=1e-3, episode_time=1.0, amp=7e-3, noise=0.05e-3):
        super().__init__()
        self.p = dict(P_NOMINAL if params is None else params)
        self.dt, self.amp, self.noise = dt, amp, noise
        self.max_steps = int(episode_time / dt)
        # Action: voltage as a fraction of supply (-1 = -12 V, +1 = +12 V)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        # Observation: 6 scaled values (see _obs)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(6,), dtype=np.float32)

    def _ref(self, t):
        """Reference position and velocity at time t."""
        w = 2 * np.pi * FREQ_HZ
        return self.amp * np.sin(w*t + self.phase), self.amp * w * np.cos(w*t + self.phase)

    def _obs(self):
        x_meas = self.x + self.np_random.normal(0, self.noise)     # noisy position sensor
        v_meas = (x_meas - self.prev_x_meas) / self.dt              # velocity estimate
        self.prev_x_meas = x_meas
        r, r_dot = self._ref(self.t)
        r_next, _ = self._ref(self.t + self.dt)
        obs = np.array([
            x_meas / X_MAX,               # position
            v_meas / 0.5,                 # velocity
            r / X_MAX,                    # reference position
            r_dot / 0.5,                  # reference velocity
            (r_next - x_meas) / X_MAX,    # error to the next reference point
            self.prev_action,             # last voltage command
        ], dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = self.v = self.t = 0.0
        self.k = 0
        self.prev_action = 0.0
        self.prev_x_meas = 0.0
        self.phase = self.np_random.uniform(0, 2*np.pi)   # start at a random point in the cycle
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        self.x, self.v, i = step(self.x, self.v, a * V_SUPPLY, self.p, self.dt)
        self.t += self.dt
        self.k += 1

        r, _ = self._ref(self.t)
        err = (self.x - r) / X_MAX
        hit_stop = abs(self.x) > X_MAX
        reward = (-err**2                                  # track the reference (large error punishes more than small errors)
                  - 0.01 * (a - self.prev_action)**2       # avoid jumpy voltage
                  - (1.0 if hit_stop else 0.0))            # don't hit the end stops
        self.prev_action = a

        truncated = self.k >= self.max_steps               # episode time is up
        info = {"x": self.x, "ref": r, "current": i}
        return self._obs(), reward, False, truncated, info