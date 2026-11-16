from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from voice_coil_env_dr import VoiceCoilDREnv

if __name__ == "__main__":
    env = make_vec_env(VoiceCoilDREnv, n_envs=8, seed=0)
    model = PPO(
        "MlpPolicy", env,
        n_steps=1024, batch_size=256, learning_rate=3e-4, gamma=0.99,
        verbose=1, seed=0, device="cpu",
    )
    model.learn(total_timesteps=2_000_000)
    model.save("ppo_dr")
    print("Saved ppo_dr.zip")