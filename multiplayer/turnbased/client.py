#!/usr/bin/env python3
"""
Number Guessing Game Client

This is the client for the number guessing game. It connects to the server
and provides a console interface for playing the game using the client framework.

Usage:
    python client.py -a localhost -p 5555

The client will connect to the server and guide you through:
1. Picking your secret number (1-100)
2. Taking turns guessing the other player's number
3. Receiving feedback (too high, too low, or correct)
"""

import sys
import argparse
from client_game_framework import GameClient
from number_guessing_client_logic import NumberGuessingClientLogic


def main(host: str, port: int) -> int:
    """
    Main client function - creates game client and runs the game.
    
    Args:
        host: Server host address
        port: Server port number
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Create the game logic implementation
    game_logic = NumberGuessingClientLogic()
    
    # Create the client with our game logic
    client = GameClient(game_logic)
    
    # Run the game
    return client.run_game(host, port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Number Guessing Game Client")
    parser.add_argument("-a", "--address", default="localhost", 
                       help="Server address (default: localhost)")
    parser.add_argument("-p", "--port", type=int, default=5555,
                       help="Server port (default: 5555)")
    
    args = parser.parse_args()
    
    try:
        exit_code = main(args.address, args.port)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nClient terminated by user")
        sys.exit(0)
