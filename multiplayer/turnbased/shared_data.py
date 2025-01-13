from enum import Enum


class Message:
    def __init__(self, message: str):
        self.message = message

    def GetMessage(self):
        return self.message


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


class InputType2:
    pass


class ClientPhase(Enum):
    PICKING = 0
    WAITING_FOR_SERVER = 1
    GUESSING = 2
    YOU_WIN = 3
    YOU_LOOSE = 4


class ClientGameState:
    def __init__(self, phase: ClientPhase, message: str):
        self.phase = phase
        self.message = message

    def GetMessage(self) -> str:
        return self.message

    def GetPhase(self) -> ClientPhase:
        return self.phase
