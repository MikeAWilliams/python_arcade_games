"""
Number Guessing Game Implementation

This is an example implementation of a turn-based multiplayer game using the game framework.
In this game, each player picks a secret number (1-100), then players take turns guessing
each other's numbers. The first player to guess correctly wins.

This demonstrates how to use the ServerGameLogic base class to create your own games.
"""

import random
from typing import Optional, Any
from server_game_framework import ServerGameLogic, GameState, PlayerState, TurnTransitionData
from shared_data import *


class NumberGuessingGame(ServerGameLogic):
    """
    Implementation of the number guessing game logic.
    
    Game Rules:
    1. Each player picks a secret number between 1-100
    2. Players take turns guessing each other's numbers
    3. After each guess, they're told if it's too high, too low, or correct
    4. First player to guess correctly wins
    """
    
    def setup_initial_state(self, player_state: PlayerState) -> ClientGameState:
        """
        Send initial setup message asking player to pick their secret number.
        """
        return ClientGameState(
            ClientPhase.SETUP, 
            "Welcome to the Number Guessing Game!\nPick a secret number between 1 and 100."
        )
    
    def process_setup_data(self, state: GameState, data: Any, player: int) -> Optional[Error]:
        """
        Process the player's secret number choice during setup phase.
        """
        if not isinstance(data, NumberPickData):
            return Error("Expected number pick data during setup")
        
        number = data.GetNumber()
        
        # Validate number is in range
        if number < 1 or number > 100:
            return Error(f"{number} is not in the range 1 to 100")
        
        # Store the player's secret number
        player_state = state.get_player_state(player)
        player_state.set_game_data("secret_number", number)
        player_state.set_game_data("guesses", [])  # Track their guesses
        
        return None  # Success
    
    def process_game_data(self, state: GameState, data: Any, player: int) -> TurnTransitionData:
        """
        Process a guess from the current player.
        """
        transition = TurnTransitionData()
        
        if not isinstance(data, GuessData):
            transition.set_messages("Invalid guess data", "Other player sent invalid data")
            return transition
        
        guess = data.GetNumber()
        current_player = state.get_player_state(player)
        other_player = state.get_other_player_state(player)
        
        # Add guess to current player's history
        current_guesses = current_player.get_game_data("guesses", [])
        current_guesses.append(guess)
        current_player.set_game_data("guesses", current_guesses)
        
        # Get the other player's secret number
        other_secret = other_player.get_game_data("secret_number")
        
        # Check if guess is correct
        if guess == other_secret:
            transition.set_current_player_won()
            transition.set_messages(
                f"Correct! You guessed {guess} and won!",
                f"They guessed your number {other_secret}. You lose!"
            )
        elif guess < other_secret:
            transition.set_messages(
                f"You guessed {guess}. Too low!",
                f"They guessed {guess}. You said too low."
            )
        else:  # guess > other_secret
            transition.set_messages(
                f"You guessed {guess}. Too high!",
                f"They guessed {guess}. You said too high."
            )
        
        return transition
    
    def both_players_setup_complete(self, state: GameState):
        """
        Initialize the game after both players have picked their numbers.
        Randomly choose who goes first.
        """
        # Randomly decide who goes first
        first_player = random.randint(0, 1)
        second_player = 1 - first_player
        
        state.get_player_state(first_player).set_playing()
        state.get_player_state(second_player).set_waiting()
        
        print(f"Game starting! Player {first_player} goes first.")
    
    def create_client_state(self, state: GameState, player: int, phase: ClientPhase, message: str) -> ClientGameState:
        """
        Create a ClientGameState with all the game-specific data for the client.
        """
        player_state = state.get_player_state(player)
        other_player_state = state.get_other_player_state(player)
        
        # Get player's secret number and guess history
        my_number = player_state.get_game_data("secret_number", 0)
        my_guesses = player_state.get_game_data("guesses", [])
        their_guesses = other_player_state.get_game_data("guesses", [])
        
        return ClientGameState(
            phase=phase,
            message=message,
            my_num=my_number,
            my_guesses=my_guesses,
            their_guesses=their_guesses
        )
