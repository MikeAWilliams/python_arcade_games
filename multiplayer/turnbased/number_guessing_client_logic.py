"""
Number Guessing Game Client Logic

This implements the client-side logic for the number guessing game.
It handles user input, display formatting, and game-specific UI elements.
"""

import sys
from typing import Any
from client_game_framework import ClientGameLogic
from shared_data import *
from number_guessing_data import NumberPickData, GuessData


class NumberGuessingClientLogic(ClientGameLogic):
    """
    Client-side logic for the number guessing game.
    
    Handles:
    - Getting secret number from user during setup
    - Getting guesses from user during gameplay  
    - Displaying game state and feedback
    - Formatting end game results
    """
    
    def get_number_input(self, prompt: str, min_val: int = 1, max_val: int = 100) -> int:
        """
        Get a valid number input from the user within the specified range.
        
        Args:
            prompt: Message to display to user
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Valid integer within range
        """
        while True:
            try:
                user_input = input(f"{prompt} ({min_val}-{max_val}): ")
                number = int(user_input)
                
                if min_val <= number <= max_val:
                    return number
                else:
                    print(f"Please enter a number between {min_val} and {max_val}")
                    
            except ValueError:
                print(f"'{user_input}' is not a valid number. Please try again.")
            except KeyboardInterrupt:
                print("\nGame interrupted by user")
                sys.exit(0)
    
    def get_setup_input(self, server_message: str) -> Any:
        """
        Get the player's secret number during setup phase.
        
        Args:
            server_message: Welcome message from server
            
        Returns:
            NumberPickData with the player's secret number
        """
        number = self.get_number_input("Enter your secret number")
        return NumberPickData(number)
    
    def get_game_input(self, state: ClientGameState) -> Any:
        """
        Get the player's guess during the game phase.
        
        Args:
            state: Current game state from server
            
        Returns:
            GuessData with the player's guess
        """
        guess = self.get_number_input("Enter your guess")
        return GuessData(guess)
    
    def display_game_state(self, state: ClientGameState):
        """
        Display the current game state when it's the player's turn.
        
        Args:
            state: Current game state from server
        """
        print(f"\n{state.GetMessage()}")
        print(f"Your secret number: {state.GetMyNum()}")
        print(f"Your past guesses: {state.GetMyGuesses()}")
        print(f"Their past guesses: {state.GetTheirGuesses()}")
    
    def display_waiting_state(self, state: ClientGameState):
        """
        Display waiting state when it's the other player's turn.
        
        Args:
            state: Current game state from server
        """
        print(f"\n{state.GetMessage()}")
    
    def display_game_end(self, state: ClientGameState, won: bool):
        """
        Display the final game result with detailed statistics.
        
        Args:
            state: Final game state
            won: True if player won, False if lost
        """
        print(f"\n{'='*50}")
        if won:
            print("🎉 CONGRATULATIONS! YOU WON! 🎉")
        else:
            print("😞 GAME OVER - YOU LOST 😞")
        
        print(f"\nFinal message: {state.GetMessage()}")
        print(f"Your secret number was: {state.GetMyNum()}")
        print(f"Your guesses: {state.GetMyGuesses()}")
        print(f"Their guesses: {state.GetTheirGuesses()}")
        
        # Display some game statistics
        my_guess_count = len(state.GetMyGuesses())
        their_guess_count = len(state.GetTheirGuesses())
        
        print(f"\nGame Statistics:")
        print(f"  Your total guesses: {my_guess_count}")
        print(f"  Their total guesses: {their_guess_count}")
        
        if won:
            print(f"  You won in {my_guess_count} guess{'es' if my_guess_count != 1 else ''}!")
        else:
            print(f"  They won in {their_guess_count} guess{'es' if their_guess_count != 1 else ''}!")
        
        print(f"{'='*50}")
    
    def handle_error(self, error_message: str):
        """
        Handle error messages from the server with game-specific formatting.
        
        Args:
            error_message: Error message from server
        """
        print(f"❌ Error: {error_message}")
        print("Please try again.")
