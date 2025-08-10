"""
Multiplayer Turn-Based Game Framework

This module provides the core framework for creating turn-based multiplayer games
that communicate via websockets. It handles player connections, turn management,
and basic game state synchronization.

To create a new game, inherit from GameLogic and implement the required methods.
"""

import socket
from _thread import *
import sys
from typing import List, Any, Optional
from enum import Enum
from shared_data import *
import server_network


class ServerPhase(Enum):
    """Represents the current phase of a player on the server side"""
    WAITING_FOR_CONNECTION = 0
    GAME_SETUP = 1          # Generic setup phase (was PICKING)
    SETUP_COMPLETE = 2      # Generic setup complete (was PICKED)
    PLAYING = 3             # Generic playing phase (was GUESSING)
    WAITING = 4             # Waiting for other player
    WON = 5                 # Player won
    LOST = 6                # Player lost


class PlayerState:
    """Manages the state of a single player in the game"""
    
    def __init__(self):
        self.phase = ServerPhase.WAITING_FOR_CONNECTION
        self.connection = None
        self.game_data = {}  # Generic storage for game-specific data
    
    def get_phase(self) -> ServerPhase:
        """Get the current phase of this player"""
        return self.phase
    
    def set_connection(self, connection: socket.socket):
        """Set the player's connection and move to setup phase"""
        self.connection = connection
        self.phase = ServerPhase.GAME_SETUP
    
    def get_connection(self) -> socket.socket:
        """Get the player's socket connection"""
        return self.connection
    
    def set_game_data(self, key: str, value: Any):
        """Store game-specific data for this player"""
        self.game_data[key] = value
    
    def get_game_data(self, key: str, default=None) -> Any:
        """Retrieve game-specific data for this player"""
        return self.game_data.get(key, default)
    
    def set_setup_complete(self):
        """Mark setup phase as complete"""
        self.phase = ServerPhase.SETUP_COMPLETE
    
    def set_playing(self):
        """Set player to playing phase (their turn)"""
        self.phase = ServerPhase.PLAYING
    
    def set_waiting(self):
        """Set player to waiting phase (other player's turn)"""
        self.phase = ServerPhase.WAITING
    
    def set_won(self):
        """Mark player as winner"""
        self.phase = ServerPhase.WON
    
    def set_lost(self):
        """Mark player as loser"""
        self.phase = ServerPhase.LOST


class GameState:
    """Manages the overall game state for both players"""
    
    def __init__(self):
        # Track two players
        self.player_state = [PlayerState(), PlayerState()]
        self.game_data = {}  # Global game data storage
    
    def get_player_state(self, index: int) -> PlayerState:
        """Get player state by index (0 or 1)"""
        return self.player_state[index]
    
    def get_other_player_state(self, current_index: int) -> PlayerState:
        """Get the other player's state"""
        if current_index == 0:
            return self.player_state[1]
        return self.player_state[0]
    
    def set_game_data(self, key: str, value: Any):
        """Store global game data"""
        self.game_data[key] = value
    
    def get_game_data(self, key: str, default=None) -> Any:
        """Retrieve global game data"""
        return self.game_data.get(key, default)


class TurnTransitionData:
    """Data structure for managing turn transitions and game outcomes"""
    
    def __init__(self):
        self.current_player_won = False
        self.cp_message = ""
        self.op_message = ""
    
    def set_messages(self, current_player_message: str, other_player_message: str):
        """Set messages to send to both players"""
        self.cp_message = current_player_message
        self.op_message = other_player_message
    
    def set_current_player_won(self):
        """Mark that the current player won"""
        self.current_player_won = True


class GameLogic:
    """
    Abstract base class for game-specific logic.
    Inherit from this class to implement your own turn-based game.
    """
    
    def setup_initial_state(self, player_state: PlayerState) -> ClientGameState:
        """
        Called when a player first connects. Return the initial client state.
        Override this method in your game implementation.
        """
        raise NotImplementedError("Must implement setup_initial_state")
    
    def process_setup_data(self, state: GameState, data: Any, player: int) -> Optional[Error]:
        """
        Process setup data from a player (e.g., choosing character, picking number).
        Return None if successful, Error object if there's an issue.
        Override this method in your game implementation.
        """
        raise NotImplementedError("Must implement process_setup_data")
    
    def process_game_data(self, state: GameState, data: Any, player: int) -> TurnTransitionData:
        """
        Process a game move from a player.
        Return TurnTransitionData with results and messages.
        Override this method in your game implementation.
        """
        raise NotImplementedError("Must implement process_game_data")
    
    def both_players_setup_complete(self, state: GameState):
        """
        Called when both players have completed setup.
        Use this to initialize the game and determine first player.
        Override this method in your game implementation.
        """
        raise NotImplementedError("Must implement both_players_setup_complete")
    
    def create_client_state(self, state: GameState, player: int, phase: ClientPhase, message: str) -> ClientGameState:
        """
        Create a ClientGameState for sending to the client.
        Override this method to include game-specific data.
        """
        raise NotImplementedError("Must implement create_client_state")


class GameServer:
    """
    Main game server that handles networking and delegates game logic
    """
    
    def __init__(self, game_logic: GameLogic):
        self.game_logic = game_logic
    
    def change_turn(self, current_player: PlayerState, other_player: PlayerState, 
                   transition: TurnTransitionData, state: GameState, current_player_index: int):
        """Handle turn transition between players"""
        current_player.set_waiting()
        other_player.set_playing()
        
        # Send state to current player (now waiting)
        cp_state = self.game_logic.create_client_state(
            state, current_player_index, ClientPhase.WAITING_FOR_SERVER,
            f"{transition.cp_message}\nIt's the other player's turn."
        )
        server_network.send(current_player.get_connection(), cp_state)
        
        # Send state to other player (now playing)
        other_player_index = 1 - current_player_index
        op_state = self.game_logic.create_client_state(
            state, other_player_index, ClientPhase.GUESSING,
            f"{transition.op_message}\nIt's your turn."
        )
        server_network.send(other_player.get_connection(), op_state)
    
    def end_game(self, current_player: PlayerState, other_player: PlayerState,
                transition: TurnTransitionData, state: GameState, current_player_index: int):
        """Handle game end when a player wins"""
        current_player.set_won()
        other_player.set_lost()
        
        # Send win state to current player
        cp_state = self.game_logic.create_client_state(
            state, current_player_index, ClientPhase.YOU_WIN,
            f"{transition.cp_message}\nCongratulations, you win!"
        )
        server_network.send(current_player.get_connection(), cp_state)
        
        # Send lose state to other player
        other_player_index = 1 - current_player_index
        op_state = self.game_logic.create_client_state(
            state, other_player_index, ClientPhase.YOU_LOOSE,
            f"{transition.op_message}\nYou lose!"
        )
        server_network.send(other_player.get_connection(), op_state)
    
    def change_turn_or_end_game(self, current_player: PlayerState, other_player: PlayerState,
                               transition: TurnTransitionData, state: GameState, current_player_index: int):
        """Decide whether to change turns or end the game"""
        if transition.current_player_won:
            self.end_game(current_player, other_player, transition, state, current_player_index)
        else:
            self.change_turn(current_player, other_player, transition, state, current_player_index)
    
    def notify_setup_status(self, player: int, player_state: PlayerState, 
                           other_player_state: PlayerState, state: GameState) -> bool:
        """Notify players about setup progress and return True if both are done"""
        if other_player_state.get_phase() == ServerPhase.SETUP_COMPLETE:
            # Both players have completed setup
            setup_state = self.game_logic.create_client_state(
                state, player, ClientPhase.WAITING_FOR_SERVER,
                "Both players ready. Game starting..."
            )
            server_network.send(player_state.get_connection(), setup_state)
            return True
        else:
            # Still waiting for other player
            waiting_state = self.game_logic.create_client_state(
                state, player, ClientPhase.WAITING_FOR_SERVER,
                "Waiting for other player to finish setup..."
            )
            server_network.send(player_state.get_connection(), waiting_state)
            return False
    
    def start_game_after_setup(self, state: GameState):
        """Initialize the game after both players complete setup"""
        self.game_logic.both_players_setup_complete(state)
        
        # Determine which player goes first (game logic should set this up)
        for i in range(2):
            player_state = state.get_player_state(i)
            if player_state.get_phase() == ServerPhase.PLAYING:
                # This player goes first
                first_state = self.game_logic.create_client_state(
                    state, i, ClientPhase.GUESSING, "It's your turn."
                )
                server_network.send(player_state.get_connection(), first_state)
            elif player_state.get_phase() == ServerPhase.WAITING:
                # This player waits
                wait_state = self.game_logic.create_client_state(
                    state, i, ClientPhase.WAITING_FOR_SERVER, "It's the other player's turn."
                )
                server_network.send(player_state.get_connection(), wait_state)
    
    def process_input(self, state: GameState, input_data: Any, player: int) -> Optional[Error]:
        """Process input from a player based on their current phase"""
        player_state = state.get_player_state(player)
        
        if player_state.get_phase() == ServerPhase.GAME_SETUP:
            # Handle setup phase
            error = self.game_logic.process_setup_data(state, input_data, player)
            if error is None:
                player_state.set_setup_complete()
                other_player_state = state.get_other_player_state(player)
                both_ready = self.notify_setup_status(player, player_state, other_player_state, state)
                if both_ready:
                    self.start_game_after_setup(state)
            return error
            
        elif player_state.get_phase() == ServerPhase.PLAYING:
            # Handle game move
            other_player = state.get_other_player_state(player)
            transition = self.game_logic.process_game_data(state, input_data, player)
            self.change_turn_or_end_game(player_state, other_player, transition, state, player)
            return None
            
        else:
            return Error(f"Received input during invalid phase: {player_state.get_phase()}")
    
    def game_loop(self, connection, state: GameState, player: int):
        """Main game loop for a single player connection"""
        player_state = state.get_player_state(player)
        player_state.set_connection(connection)
        
        # Send initial setup state
        initial_state = self.game_logic.setup_initial_state(player_state)
        server_network.send(connection, initial_state)
        
        while True:
            try:
                input_data = server_network.recieve(connection)
                
                if not input_data:
                    print(f"Player {player} disconnected")
                    break
                
                error = self.process_input(state, input_data, player)
                
                if error is not None:
                    server_network.send(connection, error)
                    
            except Exception as e:
                print(f"Error in game loop for player {player}: {e}")
                break
        
        print(f"Lost connection to player {player}")
        connection.close()
    
    def start_server(self, host: str, port: int, timeout: int = 10) -> int:
        """Start the game server and wait for connections"""
        main_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        main_socket.settimeout(timeout)
        
        try:
            main_socket.bind((host, port))
        except socket.error as e:
            print(f"Failed to bind socket: {e}")
            return 1
        
        main_socket.listen(2)
        print("Waiting for players to connect...")
        
        game_state = GameState()
        current_player = 0
        
        while current_player < 2:
            try:
                connection, addr = main_socket.accept()
                print(f"Player {current_player} connected from: {addr}")
                
                start_new_thread(self.game_loop, (connection, game_state, current_player))
                current_player += 1
            except socket.timeout:
                print("Timeout waiting for connections")
                return 1
        
        print("Both players connected. Game in progress...")
        
        # Keep server alive
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("Server shutting down...")
            return 0
