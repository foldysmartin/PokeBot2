import keyboard

from pokebot_env import PokeBotEnv

env = PokeBotEnv(False, step_limit=2)
env.reset()

actions =[]

keyboard.add_hotkey('a', lambda: actions.append(0))
# keyboard.add_hotkey('s', lambda: actions.append(1))
keyboard.add_hotkey('left', lambda: actions.append(1))
keyboard.add_hotkey('right', lambda: actions.append(2))
keyboard.add_hotkey('up', lambda: actions.append(3))
keyboard.add_hotkey('down', lambda: actions.append(4))
# keyboard.add_hotkey('enter', lambda: actions.append(6))
# keyboard.add_hotkey('space', lambda: actions.append(7))

keyboard.add_hotkey('r', lambda: actions.append(8))

# keyboard.add_hotkey('s', lambda: env.step(1))
# keyboard.add_hotkey('left', lambda: env.step(2))
# keyboard.add_hotkey('right', lambda: env.step(3))
# keyboard.add_hotkey('up', lambda: env.step(4))
# keyboard.add_hotkey('down', lambda: env.step(5))
# keyboard.add_hotkey('enter', lambda: env.step(6))
# keyboard.add_hotkey('space', lambda: env.step(7))

while True:
    if len(actions) > 0:
        action = actions.pop(0)
        if action == 8:
            env.reset()
        else:

            env.step(action)
        actions = []