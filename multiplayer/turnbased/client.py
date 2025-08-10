#!/usr/bin/env python3
"""
Number Guessing Game Client

This is the client for the number guessing game. It connects to the server
and provides a console interface for playing the game.

Usage:
    python client.py -a localhost -p 5555

The client will connect to the server and guide you through:
1. Picking your secret number (1-100)
2. Taking turns guessing the other player's number
3. Receiving feedback (too high, too low, or correct)
"""

import sys
import argparse
from client_network import ClientNetwork
from shared_data import *


def handle_connection_data(data) -> ClientGameState:
    """
    Handle the initial connection response from the server.
    
    Args:
        data: Response from server after connection
        
    Returns:
        ClientGameState if successful
        
    Raises:
        Exception: If connection failed or received unexpected data
    """
    match data:
        case ClientGameState() as state:
            return state
        case Error() as error:
            print(f"Connection error: {error.GetMessage()}")
            raise Exception(error.GetMessage())
        case _:
            print("Received unknown response type during connection")
            raise Exception("Unknown error occurred during connection")


def get_number_input(prompt: str, min_val: int = 1, max_val: int = 100) -> int:
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


def handle_number_picking_phase(coms: ClientNetwork, server_message: str) -> ClientGameState:
    """
    Handle the number picking phase of the game.
    
    Args:
        coms: Network communication object
        server_message: Initial message from server
        
    Returns:
        ClientGameState after number is picked
        
    Raises:
        Exception: If server returns unexpected response
    """
    while True:
        print(f"\n{server_message}")
        number = get_number_input("Enter your secret number")
        
        # Send number to server and wait for response
        response = coms.send_recieve(NumberPickData(number))
        
        match response:
            case Error() as error:
                print(f"Server error: {error.GetMessage()}")
                # Continue loop to let user try again
            case ClientGameState() as new_state:
                return new_state
            case _:
                raise Exception("Received unexpected response from server during number picking")


def handle_guessing_phase(coms: ClientNetwork, state: ClientGameState):
    """
    Handle the guessing phase where player makes a guess.
    
    Args:
        coms: Network communication object
        state: Current game state from server
    """
    print(f"\n{state.GetMessage()}")
    print(f"Your secret number: {state.GetMyNum()}")
    print(f"Your past guesses: {state.GetMyGuesses()}")
    print(f"Their past guesses: {state.GetTheirGuesses()}")
    
    guess = get_number_input("Enter your guess")
    coms.send(GuessData(guess))


def display_game_end(state: ClientGameState, won: bool):
    """
    Display the final game result.
    
    Args:
        state: Final game state
        won: True if player won, False if lost
    """
    print(f"\n{'='*50}")
    if won:
        print(" CONGRATULATIONS! YOU WON! ")
    else:
        print(" GAME OVER - YOU LOST ")
    
    print(f"\nFinal message: {state.GetMessage()}")
    print(f"Your secret number was: {state.GetMyNum()}")
    print(f"Your guesses: {state.GetMyGuesses()}")
    print(f"Their guesses: {state.GetTheirGuesses()}")
    print(f"{'='*50}")


def main(host: str, port: int) -> int:
    """
    Main client function - connects to server and runs the game.
    
    Args:
        host: Server host address
        port: Server port number
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    print(f"Connecting to Number Guessing Game server at {host}:{port}...")
    
    try:
        # Establish connection
        coms = ClientNetwork(host, port)
        initial_state = handle_connection_data(coms.connect())
        
        # Verify we're in the expected initial state
        if initial_state.GetPhase() != ClientPhase.PICKING:
            raise Exception(f"Unexpected initial phase: {initial_state.GetPhase()}")
        
        print("Connected successfully!")
        
        # Phase 1: Pick secret number
        post_pick_state = handle_number_picking_phase(coms, initial_state.GetMessage())
        print(f"\n{post_pick_state.GetMessage()}")
        
        # Phase 2: Game loop
        while True:
            try:
                response = coms.recieve()
                
                match response:
                    case Error() as error:
                        print(f"Server error: {error.GetMessage()}")
                        continue
                        
                    case ClientGameState() as new_state:
                        match new_state.GetPhase():
                            case ClientPhase.YOU_WIN:
                                display_game_end(new_state, won=True)
                                return 0
                                 
                            case ClientPhase.YOU_LOOSE:
                                display_game_end(new_state, won=False)
                                return 0
                                 
                            case ClientPhase.GUESSING:
                                handle_guessing_phase(coms, new_state)
                                 
                            case ClientPhase.WAITING_FOR_SERVER:
                                print(f"\n{new_state.GetMessage()}")
                                 
                            case _:
                                raise Exception(f"Unexpected game phase: {new_state.GetPhase()}")
                    
                    case _:
                        raise Exception("Received unexpected response type from server")
                        
            except KeyboardInterrupt:
                print("\nGame interrupted by user")
                return 0
                
    except Exception as e:
        print(f"Client error: {e}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Connect to a Number Guessing Game server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""" 
Examples:
    python client.py -a localhost -p 5555
    python client.py -a 192.168.1.100 -p 8080
        """
    )
    
    parser.add_argument(
        "--address", "-a",
        required=True,
        type=str,
        help="Server host address to connect to"
    )
    
    parser.add_argument(
        "--port", "-p",
        required=True,
        type=int,
        help="Server port number to connect to"
    )
    
    args = parser.parse_args()
    
    print(f"Client configuration: host={args.address}, port={args.port}")
    
    try:
        exit_code = main(args.address, args.port)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nClient interrupted by user")
        sys.exit(0)
