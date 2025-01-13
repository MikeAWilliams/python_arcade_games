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

    def GetPlayerNumber(self) -> int:
        return self.player_number

    def GetNumber(self) -> int:
        return self.number


class InputType2:
    pass


class ClientPhase(Enum):
    PICKING = 0
    WAITING_FOR_OTHER_CONNECT = 1
    WAITING_FOR_OTHER_PICK = 2


class ClientGameState:
    def __init__(self, phase: ClientPhase, message: str):
        self.phase = phase
        self.message = message

    def GetMessage(self) -> str:
        return self.message

    def GetPhase(self) -> ClientPhase:
        return self.phase
