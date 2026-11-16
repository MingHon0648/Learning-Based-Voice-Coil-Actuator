from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from voice_coil_env import VoiceCoilEnv

if __name__ == "__main__":
    # 8 copies of the environment collect experience in parallel
    env = make_vec_env(VoiceCoilEnv, n_envs=8, seed=0)

    model = PPO(
        "MlpPolicy", env,
        n_steps=1024,          # steps per env before each update (8 x 1024 = 8192 per batch)
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,            # how far ahead the agent cares (~100 steps = 0.1 s)
        verbose=1,
        seed=0,
        device="cpu",          # small networks train faster on CPU
    )
    model.learn(total_timesteps=1_000_000)
    model.save("ppo_nominal")
    print("Saved ppo_nominal.zip")