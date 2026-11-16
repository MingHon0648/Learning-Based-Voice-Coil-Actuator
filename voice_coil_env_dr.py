from actuator_model import P_NOMINAL
from voice_coil_env import VoiceCoilEnv

# Each parameter is multiplied by a random factor in this range at the start of every episode
DR_RANGES = {
    "m_a": (0.5, 1.5),
    "c_d": (0.5, 3.0),
    "b":   (0.3, 3.0),
    "Kf":  (0.8, 1.2),
    "R":   (0.85, 1.25),
}

class VoiceCoilDREnv(VoiceCoilEnv):
    """Same task as VoiceCoilEnv, but the actuator's physics change every episode."""

    def __init__(self, ranges=DR_RANGES, **kwargs):
        super().__init__(**kwargs)
        self.ranges = ranges

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        p = dict(P_NOMINAL)
        for name, (lo, hi) in self.ranges.items():
            p[name] = P_NOMINAL[name] * self.np_random.uniform(lo, hi)
        p["Kb"] = p["Kf"]        # back-EMF constant equals force constant in SI units
        self.p = p
        return obs, dict(info, params=p)