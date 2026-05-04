from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_optimised import PortfolioEnv

def make_env():
    return PortfolioEnv(split="train")

env = DummyVecEnv([make_env])

# NORMALIZATION (CRITICAL)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,  # higher exploration
    verbose=1,
)

model.learn(total_timesteps=200000)

model.save("../models/ppo_optimised")
env.save("../models/vecnorm_optimised.pkl")