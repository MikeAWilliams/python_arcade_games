"""
Shared game execution library for running Asteroids game instances.

This module provides the core game execution logic used by both benchmarking
and genetic algorithm scripts. It handles AI creation, game loop execution,
and result collection.
"""

import os
import random
from typing import Optional

import numpy as np

from asteroids.core.game import Action, Game
from asteroids.ai.heuristic import HeuristicAIInput, HeuristicAIInputParameters
from asteroids.ai.neural import NNAIInputMethod, NNAIParameters, compute_state


class GameRunner:
    """Runs a single game instance without rendering"""

    def __init__(
        self,
        width: int,
        height: int,
        ai_type: str,
        game_id: int,
        seed: Optional[int] = None,
        ai_params=None,
        record_base_name: Optional[str] = None,
    ):
        self.width = width
        self.height = height
        self.ai_type = ai_type
        self.game_id = game_id
        self.seed = seed
        self.ai_params = ai_params
        self.record_base_name = record_base_name

    def run(self) -> dict:
        """
        Run game until completion, return statistics.
        If record_base_name is set, writes recordings to ./data/{base_name}_{game_id}.npz

        Returns:
            {
                'game_id': int,
                'score': int,
                'time_alive': float
            }
        """
        # Set random seed if provided
        if self.seed is not None:
            random.seed(self.seed + self.game_id)

        # Create game instance
        game = Game(self.width, self.height)

        # Create AI input based on type
        input_method = create_ai_input(self.ai_type, game, self.ai_params)

        # Game loop with fixed timestep
        dt = 1.0 / 60.0  # 60 FPS equivalent
        tick_num = 0
        recordings = [] if self.record_base_name else None

        while game.player_alive:
            # Clear turn and acceleration every frame
            game.clear_turn()
            game.clear_acc()

            # Record state before action if recording is enabled
            if self.record_base_name:
                state = compute_state(game)

            # Get and execute action
            action = input_method.get_move()
            execute_action(game, action)

            # Record the tick data if recording is enabled
            if self.record_base_name:
                recordings.append(
                    {
                        "game_id": self.game_id,
                        "tick_num": tick_num,
                        "state": state,
                        "action": action.value,
                    }
                )

            # Update game state
            game.update(dt)
            tick_num += 1

        # Write recordings to individual file if recording was enabled
        if self.record_base_name and recordings:
            self._save_recordings(recordings)

        result = {
            "game_id": self.game_id,
            "score": game.player_score,
            "time_alive": game.time_alive,
        }

        return result

    def _save_recordings(self, recordings):
        """Save recordings to individual NPZ file"""
        filename = f"./data/{self.record_base_name}_{self.game_id}.npz"

        # Convert to numpy arrays
        game_ids = np.array([rec["game_id"] for rec in recordings], dtype=np.int32)
        tick_nums = np.array([rec["tick_num"] for rec in recordings], dtype=np.int32)
        states = np.array([rec["state"] for rec in recordings], dtype=np.float32)
        actions = np.array([rec["action"] for rec in recordings], dtype=np.int8)

        # Save compressed NPZ file
        np.savez_compressed(
            filename,
            game_ids=game_ids,
            tick_nums=tick_nums,
            states=states,
            actions=actions,
        )


def create_ai_input(ai_type: str, game, ai_params=None):
    """
    Factory function to create AI input method.

    Args:
        ai_type: "heuristic", "neural", or future AI types
        game: Game instance
        ai_params: Optional parameters (HeuristicAIInputParameters, NNAIParameters, etc.)

    Returns:
        InputMethod instance
    """
    if ai_type == "heuristic":
        params = ai_params if ai_params else HeuristicAIInputParameters()
        return HeuristicAIInput(game, params)
    elif ai_type == "neural":
        params = ai_params if ai_params else NNAIParameters()
        return NNAIInputMethod(game, params)
    else:
        raise ValueError(f"Unknown AI type: {ai_type}")


def execute_action(game, action):
    """Execute action on game"""
    if action == Action.TURN_LEFT:
        game.turning_left()
    elif action == Action.TURN_RIGHT:
        game.turning_right()
    elif action == Action.ACCELERATE:
        game.accelerate()
    elif action == Action.DECELERATE:
        game.decelerate()
    elif action == Action.SHOOT:
        game.shoot()
    elif action == Action.NO_ACTION:
        game.no_action()


def run_single_game(args) -> dict:
    """
    Run a single game (for process pool).

    Args:
        args: Tuple of (game_id, width, height, ai_type, seed, ai_params, record_base_name)

    Returns:
        Game result dictionary
    """
    game_id, width, height, ai_type, seed, ai_params, record_base_name = args
    runner = GameRunner(
        width, height, ai_type, game_id, seed, ai_params, record_base_name
    )
    return runner.run()
