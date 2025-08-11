"""
Number Guessing Game Data Classes

This file contains data classes specific to the number guessing game.
These classes define the message types sent between client and server
for game-specific actions like picking numbers and making guesses.
"""


class NumberPickData:
    """
    Data class for sending a player's secret number choice to the server.
    Used during the setup phase of the number guessing game.
    """
    def __init__(self, number: int):
        self.number = number

    def GetNumber(self) -> int:
        return self.number


class GuessData:
    """
    Data class for sending a player's guess to the server.
    Used during the playing phase of the number guessing game.
    """
    def __init__(self, number: int):
        self.number = number

    def GetNumber(self) -> int:
        return self.number
