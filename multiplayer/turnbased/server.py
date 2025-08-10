#!/usr/bin/env python3
"""
Number Guessing Game Server

This is the main server entry point for the number guessing game.
It uses the game framework to handle networking and delegates game-specific
logic to the NumberGuessingGame class.

Usage:
    python server.py -a localhost -p 5555

The server will wait for exactly 2 players to connect before starting the game.
"""

import argparse
import sys
from server_game_framework import GameServer
from number_guessing_server_logic import NumberGuessingGame


def main(host: str, port: int, timeout: int) -> int:
    """
    Start the number guessing game server.
    
    Args:
        host: Host address to bind to (e.g., 'localhost', '0.0.0.0')
        port: Port number to listen on
        timeout: Timeout in seconds for waiting for connections
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    print(f"Starting Number Guessing Game Server on {host}:{port}")
    print(f"Connection timeout: {timeout} seconds")
    print("Waiting for 2 players to connect...")
    
    # Create the game logic implementation
    game_logic = NumberGuessingGame()
    
    # Create the server with our game logic
    server = GameServer(game_logic)
    
    # Start the server
    return server.start_server(host, port, timeout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start a Number Guessing Game server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python server.py -a localhost -p 5555
    python server.py -a 0.0.0.0 -p 8080 -t 30
        """
    )
    
    parser.add_argument(
        "--address", "-a", 
        required=True, 
        type=str, 
        help="Host address to bind to (e.g., localhost, 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", "-p", 
        required=True, 
        type=int, 
        help="Port number to listen on"
    )
    
    parser.add_argument(
        "--timeout", "-t",
        required=False,
        default=10,
        type=int,
        help="Timeout for socket.accept in seconds (default: 10)"
    )
    
    args = parser.parse_args()
    
    print(f"Server configuration: host={args.address}, port={args.port}, timeout={args.timeout}")
    
    try:
        exit_code = main(args.address, args.port, args.timeout)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nServer interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)
