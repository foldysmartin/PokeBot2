from enum import Enum
import os
from pathlib import Path
from pyboy import PyBoy
from gymnasium import Env, spaces
import numpy as np

ACTIONS = ["a", "b", "left", "right", "up", "down", "start", "select"]
ACTION_SPACE = spaces.Discrete(len(ACTIONS))

directory = os.path.dirname(os.path.abspath(__file__))+"/"

class Events(Enum):
    GET_STARTER = (0xD74B, 2)
    GOT_OAKS_PARCEL = (0xD74E, 1)

class PokeBotEnv(Env):

    def __init__(self, headless=True, step_limit=1000):
        self.headless = headless
        self.screen_output_shape = (144, 160, 1)
        self.step_limit = step_limit

        obs_dict = {
            "screen": spaces.Box(low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8),
            "visited_mask": spaces.Box(
                 low=0, high=1, shape=(100,100,100), dtype=np.uint8
            ),
            "events": spaces.MultiBinary(len(Events)),
        }
        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = ACTION_SPACE

        self.pyboy = PyBoy(directory + "/pokemon_red.gb", debug=False, no_input=False, window="null" if self.headless else "SDL2",
            log_level="CRITICAL",sound_emulated=False, symbols=directory+"/pokered.sym")
        if not self.headless:
            self.pyboy.set_emulation_speed(6)
        self.screen = self.pyboy.screen

    def _get_obs(self):
        return {
            "screen": np.expand_dims(self.screen.ndarray[:, :, 1], axis=-1),
            "visited_mask": self.visited,
            "events": self._completed_events()
        }

    def _completed_events(self):
        return np.array([self.read_m(event.value[0]) for event in Events])
        

    def reset(self, seed=None):
        infos = {}
        self.steps = 0
        
        state_path =  Path(directory + '/states/inprogress.state')

        if state_path.exists():
            with open(state_path, 'rb') as f:
                self.pyboy.load_state(f)
            self.visited = np.load(directory+ '/states/visited.npy')
        else:
            with open(directory+"/states/start.state", 'rb') as f:
                self.pyboy.load_state(f)
            self.visited = np.zeros((100,100,100), dtype=np.uint8)

        self.current_reward = self._current_reward()
        self.previous_event_count = self._completed_events().sum()
        self.intiate_statistics()
        return self._get_obs(), infos
    
    def _current_reward(self):
        return self.visited.sum() * 0.012 + self._completed_events().sum() * 4
    
    def step_reward(self):
        prev_reward = self.current_reward
        self.current_reward = self._current_reward()
        return self.current_reward - prev_reward
    
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
        self.update_visited()
        self.steps += 1

        if self.previous_event_count != self._completed_events().sum():
            self.previous_event_count = self._completed_events().sum()

            if self.previous_event_count == len(Events):
                self.delete_state()
                terminal = True   
            else:                
                self.save_state()
        
        if self.steps >= self.step_limit:
            terminal = True

        if terminal:
            self.save_statistics()

        return self._get_obs(), self.step_reward(), terminal, False, {}
    
    def update_visited(self):
        x,y,map = (self.read_m("wXCoord"), self.read_m("wYCoord"), self.read_m("wCurMap"))
        self.visited[map,x,y] = 1

    
    def read_m(self, addr: str | int) -> int:
        if isinstance(addr, str):
            return self.pyboy.memory[self.pyboy.symbol_lookup(addr)[1]]
        return self.pyboy.memory[addr]
    
    def save_state(self):
        with open(directory+'states/inprogress.state', 'wb') as f:
            self.pyboy.save_state(f)

        np.save(directory+'states/visited.npy', self.visited)

    def delete_state(self):
        os.remove(directory+'states/inprogress.state')
        os.remove(directory+'states/visited.npy')
    
    def event_completed(self, event: Events) -> bool:
        return bin(256 + self.pyboy.memory[event[0]])[-event[1] - 1] == "1"

    def intiate_statistics(self):
        self.starting_stats = {
            "visited": self.visited.sum(),
            "events": self._completed_events().sum()
        }

    def save_statistics(self):
        final_stats = {
            "starting_events": self.starting_stats["events"],
            "new_exploration": self.visited.sum() - self.starting_stats["visited"],
            "events": self._completed_events().sum() - self.starting_stats["events"]
        }

        with open(directory+'statistics.txt', 'a') as f:
            # Write statistics to file as a new line comma separated values only
            f.write(','.join(map(str, final_stats.values())) + '\n')