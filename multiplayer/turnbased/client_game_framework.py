"""
Client-Side Game Framework

This module provides the client-side framework for turn-based multiplayer games.
It handles UI, input validation, and game state management on the client side.

To create a new game client, inherit from ClientGameLogic and implement the required methods.
"""

import sys
from typing import Any, Optional
from client_network import ClientNetwork
from shared_data import *


class ClientGameLogic:
    """
    Abstract base class for client-side game logic.
    Inherit from this class to implement your own game client.
    """
    
    def get_setup_input(self, server_message: str) -> Any:
        """
        Get setup input from the user (e.g., character selection, number picking).
        Override this method in your game implementation.
        
        Args:
            server_message: Message from server about what to input
            
        Returns:
            Data object to send to server
        """
        raise NotImplementedError("Must implement get_setup_input")
    
    def get_game_input(self, state: ClientGameState) -> Any:
        """
        Get game move input from the user.
        Override this method in your game implementation.
        
        Args:
            state: Current game state from server
            
        Returns:
            Data object to send to server
        """
        raise NotImplementedError("Must implement get_game_input")
    
    def display_game_state(self, state: ClientGameState):
        """
        Display the current game state to the user.
        Override this method in your game implementation.
        
        Args:
            state: Current game state from server
        """
        raise NotImplementedError("Must implement display_game_state")
    
    def display_waiting_state(self, state: ClientGameState):
        """
        Display waiting state to the user.
        Override this method in your game implementation.
        
        Args:
            state: Current game state from server
        """
        raise NotImplementedError("Must implement display_waiting_state")
    
    def display_game_end(self, state: ClientGameState, won: bool):
        """
        Display the final game result.
        Override this method in your game implementation.
        
        Args:
            state: Final game state
            won: True if player won, False if lost
        """
        raise NotImplementedError("Must implement display_game_end")
    
    def handle_error(self, error_message: str):
        """
        Handle error messages from the server.
        Override this method for custom error handling.
        
        Args:
            error_message: Error message from server
        """
        print(f"Server error: {error_message}")


class GameClient:
    """
    Main game client that handles networking and delegates UI/input to game logic
    """
    
    def __init__(self, game_logic: ClientGameLogic):
        self.game_logic = game_logic
    
    def handle_connection_data(self, data: ClientGameState|Error) -> ClientGameState:
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
    
    def handle_setup_phase(self, coms: ClientNetwork, server_message: str) -> ClientGameState:
        """
        Handle the setup phase of the game.
        
        Args:
            coms: Network communication object
            server_message: Initial message from server
            
        Returns:
            ClientGameState after setup is complete
            
        Raises:
            Exception: If server returns unexpected response
        """
        while True:
            print(f"\n{server_message}")
            
            try:
                setup_data = self.game_logic.get_setup_input(server_message)
                
                # Send setup data to server and wait for response
                response = coms.send_recieve(setup_data)
                
                match response:
                    case Error() as error:
                        self.game_logic.handle_error(error.GetMessage())
                        # Continue loop to let user try again
                    case ClientGameState() as new_state:
                        return new_state
                    case _:
                        raise Exception("Received unexpected response from server during setup")
                        
            except KeyboardInterrupt:
                print("\nGame interrupted by user")
                sys.exit(0)
            except Exception as e:
                print(f"Setup error: {e}")
                # Continue loop to let user try again
    
    def run_game(self, host: str, port: int) -> int:
        """
        Main client function - connects to server and runs the game.
        
        Args:
            host: Server host address
            port: Server port number
            
        Returns:
            Exit code (0 for success, 1 for error)
        """
        print(f"Connecting to game server at {host}:{port}...")
        
        try:
            # Establish connection
            coms = ClientNetwork(host, port)
            initial_state = self.handle_connection_data(coms.connect())
            
            # Verify we're in the expected initial state
            if initial_state.GetPhase() != ClientPhase.SETUP:
                raise Exception(f"Unexpected initial phase: {initial_state.GetPhase()}")
            
            print("Connected successfully!")
            
            post_setup_state = self.handle_setup_phase(coms, initial_state.GetMessage())
            print(f"\n{post_setup_state.GetMessage()}")
            
            # Game loop
            while True:
                try:
                    response = coms.recieve()
                    
                    match response:
                        case Error() as error:
                            self.game_logic.handle_error(error.GetMessage())
                            continue
                            
                        case ClientGameState() as new_state:
                            match new_state.GetPhase():
                                case ClientPhase.YOU_WIN:
                                    self.game_logic.display_game_end(new_state, won=True)
                                    return 0
                                    
                                case ClientPhase.YOU_LOOSE:
                                    self.game_logic.display_game_end(new_state, won=False)
                                    return 0
                                    
                                case ClientPhase.PLAYING:
                                    self.game_logic.display_game_state(new_state)
                                    game_input = self.game_logic.get_game_input(new_state)
                                    coms.send(game_input)
                                    
                                case ClientPhase.WAITING_FOR_SERVER:
                                    self.game_logic.display_waiting_state(new_state)
                                    
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
