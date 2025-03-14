from events import Events


class EventGoal:
    def __init__(self, event):
        self.event = event
        self.completed = False

    def is_completed(self, pyboy):
        if self.completed:
            print(f"EventGoal {self.event} is completed")
            return True
        
        if self.event.completed(pyboy):
            self.completed = True
            return True
        
        return False
    
class MapGoal:
    def __init__(self, map):
        self.map = map
        self.completed = False

    def is_completed(self, pyboy):
        if self.completed:
            print(f"MapGoal {self.map} is completed")
            return True
        
        if read_m(pyboy, "wCurMap") == self.map.value:
            self.completed = True
            return True
        
        return False
    
# Is duplicated in pokebot_env.py
def read_m(pyboy, addr: str | int) -> int:
        if isinstance(addr, str):
            return pyboy.memory[pyboy.symbol_lookup(addr)[1]]
        return pyboy.memory[addr]

