from pokebot_env import PokeBotEnv


ep_length = 3200

env = PokeBotEnv(False, step_limit=ep_length)
env.reset()

stop_at = 19051

# read actions from file
with open("actions/actions-5.txt", "r") as f:
    steps = 0
    for line in f:
        action = int(line.strip())
        env.step(action)

        steps += 1

        if steps >= stop_at:
            pass