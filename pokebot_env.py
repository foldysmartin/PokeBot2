from dataclasses import dataclass
import datetime
from enum import Enum
import os
from pathlib import Path
from pyboy import PyBoy
from gymnasium import Env, spaces
import numpy as np
from events import Events
from goals import EventGoal, MapGoal
from maps import Maps

ACTIONS = ["a", "left", "right", "up", "down"]
ACTION_SPACE = spaces.Discrete(len(ACTIONS))

directory = os.path.dirname(os.path.abspath(__file__))+"/"





class StuckError(Exception):
    pass


class DialogState(Enum):
    NO_DIALOG = 0
    MENU = 1
    POKEMON = 2
    ITEM = 3
    PLAYER = 4
    SAVE = 5
    OPTIONS = 6
    YESNO = 7
    SKIPPING = 8
    BATTLE_MENU = 9
    BATTLE_MOVE = 10
    BATTLE_POKEMON = 11
    BATTLE_ITEM = 12

# Add pokedex
class MenuPosition(Enum):
    POKEMON = 0
    ITEM = 1
    PLAYER = 2
    SAVE = 3
    OPTIONS= 4
    EXIT = 5

class YesNo(Enum):
    Yes = 0
    No = 1

class BattleMenuSelection(Enum):
    FIGHT = 0
    POKEMON = 1
    ITEM = 2
    RUN = 3

class MoveSelection(Enum):
    MOVE1 = 0
    MOVE2 = 1
    MOVE3 = 2
    MOVE4 = 3

@dataclass(frozen=True)
class Position:
    x: int
    y: int
    map: int

action_length = 2
tick_length = 40

def clamp(n, enum):
    return enum(max(0, min(n, len(enum) - 1)))

class PokeBotEnv(Env):

    def __init__(self, headless=True, step_limit=1000, id=0):
        print(f"Starting environment with id {id}")
        self.id = id
        self.run_id = 0
        self.headless = headless
        self.step_limit = step_limit

        obs_dict = {
            "map": spaces.Discrete(100),
            "x": spaces.Discrete(100),
            "y": spaces.Discrete(100),
            "direction": spaces.Discrete(4),
            "events": spaces.MultiDiscrete([2 for _ in Events]),
        }
        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = ACTION_SPACE

        self.pyboy = PyBoy(directory + "/pokemon_red.gb", debug=False, no_input=False, window="null" if self.headless else "SDL2",
            log_level="CRITICAL",sound_emulated=False, symbols=directory+"/pokered.sym")
        if not self.headless:
            self.pyboy.set_emulation_speed(6)
        self.screen = self.pyboy.screen

    def _get_obs(self):
        self.log_to_file("Getting observation")
        return {
            "map": self.read_m("wCurMap"),
            "x": self.read_m("wXCoord"),
            "y": self.read_m("wYCoord"),
            "direction": self.read_m("wSpritePlayerStateData1FacingDirection") // 4,            
            "events": self._completed_events(),
        }

    def _completed_events(self):
        self.log_to_file("Getting completed events")
        return np.array([self.event_completed(event) for event in Events])
        

    def reset(self, seed=None):
        print("Resetting environment")
        infos = {}
        self.steps = 0
        self.run_id += 1
        
        self.state = DialogState.NO_DIALOG
        self.yes_no = YesNo.Yes
        self.menu_position = MenuPosition.POKEMON
        self.battle_menu_selection = BattleMenuSelection.FIGHT
        self.battle_move_selecton = MoveSelection.MOVE1

        self.goals = [
                    MapGoal(Maps.PALLET_TOWN),
                    MapGoal(Maps.RED_DOWNSTAIRS),
                    EventGoal(Events.OAK_APPEARS),
                    EventGoal(Events.GET_STARTER),
                ]
        
        self.positions = []
        state_path =  Path(directory + '/states/inprogress.state')

        if state_path.exists():
            with open(state_path, 'rb') as f:
                self.pyboy.load_state(f)
        else:
            with open(directory+"/states/start.state", 'rb') as f:
                self.pyboy.load_state(f)
                
        self.current_reward = self._current_reward()
        self.previous_event_count = self._completed_events().sum()
        self.pyboy.tick(tick_length, render=True)
        
        print("Reset complete")
        return self._get_obs(), infos
    
    def _current_reward(self):
        self.log_to_file("Getting current reward")
        # Make it print the number of events completed
        [goal.is_completed(self.pyboy) for goal in self.goals]
        return sum([goal.is_completed(self.pyboy) for goal in self.goals if type(goal) is EventGoal])*10 + len(self.positions)*0.1
    
    def step_reward(self):
        self.log_to_file("Getting step reward")
        prev_reward = self.current_reward
        self.current_reward = self._current_reward()
        reward = self.current_reward - prev_reward
        return reward
    
    def dialogue_state(self, action):
        self.log_to_file("Checking dialogue state")
        if self.state == DialogState.BATTLE_MENU:
            if action == "up":
                if self.battle_menu_selection == BattleMenuSelection.RUN:
                    self.battle_menu_selection = BattleMenuSelection.POKEMON
                elif self.battle_menu_selection == BattleMenuSelection.ITEM:
                    self.battle_menu_selection = BattleMenuSelection.FIGHT
            elif action == "down":
                if  self.battle_menu_selection == BattleMenuSelection.FIGHT:
                    self.battle_menu_selection = BattleMenuSelection.ITEM
                elif self.battle_menu_selection == BattleMenuSelection.POKEMON:
                    self.battle_menu_selection = BattleMenuSelection.RUN
            elif action == "left":
                if self.battle_menu_selection == BattleMenuSelection.POKEMON:
                    self.battle_menu_selection = BattleMenuSelection.FIGHT
                elif self.battle_menu_selection == BattleMenuSelection.RUN:
                    self.battle_menu_selection = BattleMenuSelection.ITEM
            elif action == "right":
                if self.battle_menu_selection == BattleMenuSelection.FIGHT:
                    self.battle_menu_selection = BattleMenuSelection.POKEMON
                elif self.battle_menu_selection == BattleMenuSelection.ITEM:
                    self.battle_menu_selection = BattleMenuSelection.RUN
            elif action == "a":
                if self.battle_menu_selection == BattleMenuSelection.FIGHT:
                    self.state = DialogState.BATTLE_MOVE
                elif self.battle_menu_selection == BattleMenuSelection.POKEMON:
                    #go back
                    self.pyboy.button("b", action_length)
                    self.pyboy.tick(tick_length, render=True)        
                elif self.battle_menu_selection == BattleMenuSelection.ITEM:
                    #go back
                    self.pyboy.button("b", action_length)
                    self.pyboy.tick(tick_length, render=True)  
                elif self.battle_menu_selection == BattleMenuSelection.RUN:
                    self.state = DialogState.SKIPPING
        elif self.state == DialogState.BATTLE_MOVE:
            if action == "up":
                self.battle_move_selecton = MoveSelection((self.battle_move_selecton.value - 1) % len(MoveSelection))
            elif action == "down":
                self.battle_move_selecton = MoveSelection((self.battle_move_selecton.value + 1) % len(MoveSelection))
            elif action == "a":
                self.state = DialogState.SKIPPING
            elif action == "b":
                self.state = DialogState.BATTLE_MENU
            

        elif self.read_m("wCurOpponent") != 0:
            if self.read_m("wTextBoxID") == 11:
                self.state = DialogState.BATTLE_MENU
            else:
                action = "a"
                self.state = DialogState.SKIPPING
        elif self.read_m("vChars1") == 0:
            self.state = DialogState.NO_DIALOG         
            
            # Reset battke menu selection for future battles
            self.battle_menu_selection = BattleMenuSelection.FIGHT
            self.battle_move_selecton = MoveSelection.MOVE1
            
            # Reset text box id so we can skip dialog in the future
            self.set_m("wTextBoxID", 1)  
        elif self.state == DialogState.SAVE or self.state == DialogState.OPTIONS or self.state == DialogState.PLAYER:
            # Skip dialog
            action = "b"
            self.state = DialogState.SKIPPING
        elif self.state == DialogState.NO_DIALOG: # This is only called if there is a change from no dialog to dialog
            if action == "start":
                self.state = DialogState.MENU
            else:
                action = "a"
                self.state = DialogState.SKIPPING
        elif self.state == DialogState.MENU:
            if action == "up":
                self.menu_position = MenuPosition((self.menu_position.value - 1) % 6)
            elif action == "down":
                self.menu_position =  MenuPosition((self.menu_position.value + 1) % 6)
            elif action == "a":
                if not self.event_completed(Events.GET_STARTER) and self.menu_position == MenuPosition.POKEMON:
                    # No pokemon yet so cannot open menu
                    pass
                else:
                    self.state = DialogState(self.menu_position.value + DialogState.MENU.value + 1)
                    if self.state == DialogState.SAVE or self.state == DialogState.OPTIONS or self.state == DialogState.PLAYER or self.state == DialogState.POKEMON:
                        # Recursively skip dialog including ignoring pokemon for now
                        action = "b"
                        self.state = DialogState.SKIPPING
        elif self.state == DialogState.ITEM:
            if action == "b":
                self.state = DialogState.MENU
            elif action == "a":
                # Currently no items
                self.state = DialogState.MENU
        elif self.state == DialogState.YESNO:
            if action == "a":
                self.state = DialogState.SKIPPING
                self.yes_no = YesNo.No
                # Reset text box id so we can skip dialog in the future
                self.set_m("wTextBoxID", 1)  
            elif action == "up" or action == "down":
                self.yes_no = YesNo((self.yes_no.value + 1) % 2)
        elif self.state == DialogState.SKIPPING:
            # if self.read_m("wTextBoxID") == 20:
            #     self.state = DialogState.YESNO
            #     self.yes_no = YesNo.Yes

            if self.read_m("wTextBoxID") == 13:
                # In pc skip with b
                self.state = DialogState.SKIPPING
                action = "b"

        
        if self.state == DialogState.SKIPPING:            
            self.pyboy.button(action, action_length)
            self.pyboy.tick(tick_length, render=True)
            
            self.dialogue_state(action)



    def run_action_on_emulator(self, action):
        self.log_to_file(f"Running action {action}")
        self.log_action(action)
        self.pyboy.button(ACTIONS[action], action_length)
        self.pyboy.tick(tick_length, render=True)

        self.dialogue_state(ACTIONS[action])

        # DO NOT DELETE. Some animations require dialog navigation
        for _ in range(1000):
            if not self.read_m("wJoyIgnore"):
                break
            self.pyboy.button("a", action_length)
            self.pyboy.tick(tick_length, render=True)


    def step(self, action):
        self.log_to_file(f"Taking step {self.steps}")
        terminal = False

        self.run_action_on_emulator(action)
        self.steps += 1

        position = self.get_location()
        if position not in self.positions:
            self.positions.append(position)

        reward = self.step_reward()

        if self.previous_event_count != self._completed_events().sum():
            self.previous_event_count = self._completed_events().sum()
            if self.previous_event_count == len(Events):
                self.delete_state()
                print("All events completed")
                terminal = True   
            else:                
                self.save_state()
        if self.steps >= self.step_limit:
            terminal = True
            print("No new exploration")

        

        return self._get_obs(), reward, terminal, False, {}
    
    def get_location(self):
        return Position(self.read_m("wXCoord"), self.read_m("wYCoord"), self.read_m("wCurMap"))

    
    def read_m(self, addr: str | int) -> int:
        if isinstance(addr, str):
            return self.pyboy.memory[self.pyboy.symbol_lookup(addr)[1]]
        return self.pyboy.memory[addr]
    
    def set_m(self, addr: str | int, value: int):
        if isinstance(addr, str):
            self.pyboy.memory[self
                .pyboy.symbol_lookup(addr)[1]] = value
        else:
            self.pyboy.memory[addr] = value
    
    def save_state(self):
        self.log_to_file("Saving state")
        with open(directory+'states/inprogress.state', 'wb') as f:
            self.pyboy.save_state(f)

        with open(directory+'states/inprogress.txt', 'wt') as f:
            f.write(str(self.positions))

    def delete_state(self):
        self.log_to_file("Deleting state")
        os.remove(directory+'states/inprogress.state')
        os.remove(directory+'states/inprogress.txt')

    def load_state(self):
        self.log_to_file("Loading state")
        with open(directory+'states/inprogress.state', 'rb') as f:
            self.pyboy.load_state(f)
        with open(directory+'states/inprogress.txt', 'rt') as f:
            self.positions = eval(f.read())
    
    def event_completed(self, event: Events) -> bool:
        return bin(256 + self.pyboy.memory[event.value[0]])[-event.value[1] - 1] == "1"
    

    def log_to_file(self, message):
        os.makedirs(directory + "/logs", exist_ok=True)
        with open(directory + f"/logs/log-{self.run_id}-{self.id}.txt", "a") as f:
            f.write(message + "\n")

    def log_action(self, action):
        os.makedirs(directory + "/actions", exist_ok=True)
        with open(directory + f"actions/actions-{self.run_id}-{self.id}.txt", "a") as f:
            f.write(str(action) + "\n")
