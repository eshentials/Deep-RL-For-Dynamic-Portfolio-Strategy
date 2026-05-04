import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_fixed import PortfolioEnv

def evaluate():
    env = DummyVecEnv([lambda: PortfolioEnv(split="val")])
    env = VecNormalize.load("../models/vecnorm_optimised.pkl", env)
    env.training = False

    model = PPO.load("../models/ppo_optimised")

    obs = env.reset()
    done = False

    navs = [1.0]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        navs.append(env.get_attr("nav")[0])

    navs = np.array(navs)

    returns = np.diff(navs) / navs[:-1]

    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

    print("Final NAV:", navs[-1])
    print("Sharpe:", sharpe)

if __name__ == "__main__":
    evaluate()