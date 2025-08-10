# Turn-Based Multiplayer Game Framework

A Python framework for creating turn-based multiplayer games that communicate via websockets. This repository includes both a reusable framework and an example number guessing game implementation.

## 🎯 What's Included

### Framework Files (Reusable)
- **`game_framework.py`** - Core networking and game state management framework
- **`client_network.py`** - Client-side networking utilities  
- **`server_network.py`** - Server-side networking utilities
- **`shared_data.py`** - Shared data structures and message types

### Example Game Implementation
- **`number_guessing_game.py`** - Example game logic using the framework
- **`server.py`** - Server entry point for the number guessing game
- **`client.py`** - Client entry point with improved UI and error handling

## 🎮 Number Guessing Game

The included example game demonstrates how to use the framework:

1. **Setup Phase**: Each player picks a secret number (1-100)
2. **Game Phase**: Players take turns guessing each other's numbers
3. **Feedback**: After each guess, players receive "too high", "too low", or "correct"
4. **Victory**: First player to guess correctly wins!

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (uses match/case statements)
- No external dependencies required

### Running the Number Guessing Game

1. **Start the server:**
   ```cmd
   python server.py -a localhost -p 5555
   ```

2. **Connect first player (in another terminal):**
   ```cmd
   python client.py -a localhost -p 5555
   ```

3. **Connect second player (in another terminal):**
   ```cmd
   python client.py -a localhost -p 5555
   ```

### Command Line Options

The server and client support:
- `-a, --address`: Host address (e.g., `localhost`, `0.0.0.0`)
- `-p, --port`: Port number (e.g., `5555`)
- `-t, --timeout`: Connection timeout in seconds (server only, default: 10)

**Examples:**
```cmd
# Local game
python server.py -a localhost -p 5555

# Allow connections from other machines
python server.py -a 0.0.0.0 -p 8080 -t 30

# Connect to remote server
python client.py -a 192.168.1.100 -p 8080
```

## 🛠️ Creating Your Own Game

To create a new turn-based game using this framework:

### 1. Implement Game Logic

Create a new class inheriting from `GameLogic`:

```python
from game_framework import GameLogic, GameState, PlayerState, TurnTransitionData
from shared_data import *

class MyGame(GameLogic):
    def setup_initial_state(self, player_state: PlayerState) -> ClientGameState:
        # Return initial setup message for new players
        return ClientGameState(ClientPhase.PICKING, "Welcome! Choose your character...")
    
    def process_setup_data(self, state: GameState, data: Any, player: int) -> Optional[Error]:
        # Handle setup phase (character selection, etc.)
        # Return None for success, Error object for validation failures
        pass
    
    def process_game_data(self, state: GameState, data: Any, player: int) -> TurnTransitionData:
        # Handle game moves and return transition data
        transition = TurnTransitionData()
        # ... game logic here ...
        return transition
    
    def both_players_setup_complete(self, state: GameState):
        # Initialize game after both players finish setup
        # Set first player, initialize game state, etc.
        pass
    
    def create_client_state(self, state: GameState, player: int, phase: ClientPhase, message: str) -> ClientGameState:
        # Create client state with game-specific data
        return ClientGameState(phase, message, ...)
```

### 2. Create Server Entry Point

```python
from game_framework import GameServer
from my_game import MyGame

def main(host: str, port: int, timeout: int):
    game_logic = MyGame()
    server = GameServer(game_logic)
    return server.start_server(host, port, timeout)
```

### 3. Define Custom Data Types

Add your game-specific message types to `shared_data.py`:

```python
class MyGameMoveData:
    def __init__(self, move_type: str, data: dict):
        self.move_type = move_type
        self.data = data
```

### 4. Create Client Implementation

Modify the client to handle your game's specific UI and input requirements.

## 📁 Project Structure

```
multiplayer/turnbased/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── Framework Files:
├── game_framework.py         # Core framework classes
├── client_network.py         # Client networking
├── server_network.py         # Server networking  
├── shared_data.py           # Shared data structures
│
├── Example Game:
├── number_guessing_game.py   # Game logic implementation
├── server_new.py            # New server entry point
├── client_new.py            # New client with better UI
│
└── Legacy Files:
├── server.py                # Original mixed implementation
└── client.py                # Original client
```

## 🔧 Framework Architecture

### Key Components

1. **GameLogic (Abstract Base Class)**
   - Defines the interface for game-specific logic
   - Separates game rules from networking concerns

2. **GameServer**
   - Handles networking, connections, and turn management
   - Delegates game-specific decisions to GameLogic implementation

3. **PlayerState & GameState**
   - Manages player and game state with generic data storage
   - Supports any game type through flexible data structures

4. **TurnTransitionData**
   - Encapsulates turn results and player messages
   - Handles win conditions and turn changes

### Benefits of the Framework Approach

- **Separation of Concerns**: Networking code is separate from game logic
- **Reusability**: Create new games without rewriting networking code
- **Maintainability**: Clear interfaces make code easier to understand and modify
- **Extensibility**: Easy to add new features to either networking or game logic independently

## 🐛 Troubleshooting

### Common Issues

1. **"Address already in use" error**
   - Wait a few seconds and try again, or use a different port
   - Kill any existing server processes

2. **Connection timeout**
   - Check firewall settings
   - Ensure server is running before starting clients
   - Verify host/port parameters match

3. **Import errors**
   - Ensure you're running from the correct directory
   - Check that all files are present
   - Activate virtual environment if using one

### Debug Tips

- Server prints connection status and player actions
- Client shows detailed error messages
- Use `localhost` for local testing, `0.0.0.0` to allow external connections

## 🤝 Contributing

This framework is designed to be extended! Feel free to:
- Create new game implementations
- Improve the networking layer
- Add new features to the framework
- Fix bugs or improve documentation

## 📝 License

This project is provided as-is for educational and development purposes.
