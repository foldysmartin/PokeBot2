from pathlib import Path
from pyboy import PyBoy
from gymnasium import Env, spaces
import numpy as np

ACTIONS = ["a", "b", "left", "right", "up", "down", "start", "select"]
ACTION_SPACE = spaces.Discrete(len(ACTIONS))

class PokeBotEnv(Env):

    def __init__(self, headless=True, step_limit=1000):
        self.headless = headless
        self.screen_output_shape = (144, 160, 1)
        self.step_limit = step_limit

        obs_dict = {
            "screen": spaces.Box(low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8),
            # "visited_mask": spaces.Box(
            #     low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
            # ),
        }
        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = ACTION_SPACE

        self.pyboy = PyBoy("pokemon_red.gb", debug=False, no_input=False, window="null" if self.headless else "SDL2",
            log_level="CRITICAL",sound_emulated=False, symbols="pokered.sym")
        if not self.headless:
            self.pyboy.set_emulation_speed(6)
        self.screen = self.pyboy.screen

    def _get_obs(self):
        return {
            "screen": np.expand_dims(self.screen.ndarray[:, :, 1], axis=-1),
            # "visited_mask": self._get_visited_mask(),
        }
        

    def reset(self, seed=None):
        infos = {}
        self.steps = 0

        state_path =  Path('states/inprogress.state')

        if state_path.exists():
            with open(state_path, 'rb') as f:
                self.pyboy.load_state(f)
        else:
            with open("states/start.state", 'rb') as f:
                self.pyboy.load_state(f)

        return self._get_obs(), infos
    
    def run_action_on_emulator(self, action):
        action_freq = 23
        self.pyboy.button(ACTIONS[action])
        self.pyboy.tick(action_freq - 1, render=False)
        # DO NOT DELETE. Some animations require dialog navigation
        for _ in range(1000):
            if not self.read_m("wJoyIgnore"):
                break
            self.pyboy.button("a")
            self.pyboy.tick(action_freq, render=False)

        # One last tick just in case
        self.pyboy.tick(1, render=True)

    def step(self, action):
        terminal = False

        self.run_action_on_emulator(action)
        self.steps += 1
        if self.steps >= self.step_limit:
            terminal = True

        return self._get_obs(), 0, terminal, False, {}
    
    def read_m(self, addr: str | int) -> int:
        if isinstance(addr, str):
            return self.pyboy.memory[self.pyboy.symbol_lookup(addr)[1]]
        return self.pyboy.memory[addr]