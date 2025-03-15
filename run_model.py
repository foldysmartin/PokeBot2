from stable_baselines3 import PPO
from pokebot_env import PokeBotEnv

model = PPO.load("model.zip")

env = PokeBotEnv(False, step_limit=3200)
obs, _ = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info, _ = env.step(action) 