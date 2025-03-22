from enum import Enum


class Events(Enum):
    OAK_APPEARS = (0xD74B, 7)
    GET_STARTER = (0xD74B, 2)
    RIVAL_IN_LAB = (0xD74B, 3)
    GOT_OAKS_PARCEL = (0xD74E, 2)

    def completed(self, pyboy) -> bool:
        return bin(256 + pyboy.memory[self.value[0]])[-self.value[1] - 1] == "1"