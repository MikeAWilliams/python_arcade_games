from enum import Enum
from typing import List


class Error:
    def __init__(self, message):
        self.message = message

    def GetMessage(self):
        return self.message


class NumberPickData:
    def __init__(self, number: int):
        self.number = number

    def GetNumber(self) -> int:
        return self.number


class GuessData:
    def __init__(self, number: int):
        self.number = number

    def GetNumber(self) -> int:
        return self.number


class ClientPhase(Enum):
    SETUP = 0
    WAITING_FOR_SERVER = 1
    PLAYING = 2
    YOU_WIN = 3
    YOU_LOOSE = 4


class ClientGameState:
    def __init__(
        self,
        phase: ClientPhase,
        message: str,
        my_num: int = 0,
        my_guesses: List[int] = [],
        their_guesses: List[int] = [],
    ):
        self.phase = phase
        self.message = message
        self.my_num = my_num
        self.my_guesses = my_guesses
        self.their_guesses = their_guesses

    def GetMessage(self) -> str:
        return self.message

    def GetPhase(self) -> ClientPhase:
        return self.phase

    def GetMyNum(self) -> int:
        return self.my_num

    def GetMyGuesses(self) -> List[int]:
        return self.my_guesses

    def GetTheirGuesses(self) -> List[int]:
        return self.their_guesses
