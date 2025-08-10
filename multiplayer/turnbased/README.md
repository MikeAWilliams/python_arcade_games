# Turn-Based Multiplayer Game Framework

A Python framework for creating turn-based multiplayer games that communicate via websockets. This repository includes both a reusable framework and an example number guessing game implementation.

## 🎯 What's Included

### Framework Files (Reusable)
- **`server_game_framework.py`** - Server-side networking and game state management framework
- **`client_game_framework.py`** - Client-side framework with abstract game logic interface
- **`client_network.py`** - Client-side networking utilities  
- **`server_network.py`** - Server-side networking utilities
- **`shared_data.py`** - Shared data structures and message types

### Example Game Implementation
- **`number_guessing_server_logic.py`** - Server-side game logic using the framework
- **`number_guessing_client_logic.py`** - Client-side game logic implementation
- **`number_guessing_data.py`** - Game-specific data classes (NumberPickData, GuessData)
- **`server.py`** - Server entry point for the number guessing game
- **`client.py`** - Client entry point using the client framework

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

### 1. Implement Server Game Logic

Create a new class inheriting from `MyGameServerLogic(ServerGameLogic):`

### 2. Modify server.py 

    Replace ` game_logic = NumberGuessingGame()`
    with ` game_logic = MyGameServerLogic()`

### 3. Define Custom Data Types

Add your game-specific message types to `shared_data.py`:

```python
class MyGameMoveData:
    def __init__(self, move_type: str, data: dict):
        self.move_type = move_type
        self.data = data
```

### 4. Create Client Game Logic

Implement client-side game logic by inheriting from `ClientGameLogic`:

### 5. Create Client Entry Point
    
    Replace ` game_logic = NumberGuessingGame()`
    with ` game_logic = MyGameClientLogic()`
