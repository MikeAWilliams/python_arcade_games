"""
Headless multi-process Asteroids game simulation for AI benchmarking.

This module runs multiple game instances in parallel without rendering,
collecting statistics on AI performance. Uses ProcessPoolExecutor to
bypass the Python GIL and achieve true parallel execution across all CPU cores.
"""

import argparse
import logging
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from asteroids.core.game_runner import run_single_game


def setup_headless_logging(base_name):
    """
    Set up dual logging (console + file) for headless recording.
    Log file saved to data directory as <base_name>.log (overwritten each run).

    Args:
        base_name: Base name for the log file (e.g., "game_recordings" -> "data/game_recordings.log")

    Returns: logger instance
    """
    log_file = f"data/{base_name}.log"

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Configure logger
    logger = logging.getLogger("headless")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # File handler (overwrite mode)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)

    # Simple format (no logger name prefix)
    formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class StatisticsCollector:
    """Collects and analyzes results from multiple games"""

    def __init__(self):
        self.results = []

    def add_result(self, result: dict):
        self.results.append(result)

    def compute_statistics(self) -> dict:
        """
        Compute comprehensive statistics

        Returns:
            {
                'total_games': int,
                'score': {
                    'min': int,
                    'max': int,
                    'avg': float,
                    'median': float,
                    'std_dev': float
                },
                'time_alive': {
                    'min': float,
                    'max': float,
                    'avg': float,
                    'median': float
                }
            }
        """
        if not self.results:
            return {}

        scores = [r["score"] for r in self.results]
        times = [r["time_alive"] for r in self.results]

        return {
            "total_games": len(self.results),
            "score": {
                "min": min(scores),
                "max": max(scores),
                "avg": statistics.mean(scores),
                "median": statistics.median(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            },
            "time_alive": {
                "min": min(times),
                "max": max(times),
                "avg": statistics.mean(times),
                "median": statistics.median(times),
            },
        }

    def print_summary(self, logger=None):
        """Print formatted statistics summary"""
        stats = self.compute_statistics()

        log = logger.info if logger else print

        log("\n" + "=" * 60)
        log("GAME STATISTICS SUMMARY")
        log("=" * 60)
        log(f"Total Games: {stats['total_games']}")
        log("")
        log("SCORE:")
        log(f"  Min:       {stats['score']['min']:>6}")
        log(f"  Max:       {stats['score']['max']:>6}")
        log(f"  Average:   {stats['score']['avg']:>6.1f}")
        log(f"  Median:    {stats['score']['median']:>6.1f}")
        log(f"  Std Dev:   {stats['score']['std_dev']:>6.1f}")
        log("")
        log("SURVIVAL TIME (seconds):")
        log(f"  Min:       {stats['time_alive']['min']:>6.2f}")
        log(f"  Max:       {stats['time_alive']['max']:>6.2f}")
        log(f"  Average:   {stats['time_alive']['avg']:>6.2f}")
        log(f"  Median:    {stats['time_alive']['median']:>6.2f}")
        log("=" * 60)


def run_parallel_games(
    num_games: int,
    num_threads: int,
    width: int,
    height: int,
    ai_type: str,
    seed: Optional[int] = None,
    show_progress: bool = True,
    model_path: Optional[str] = None,
    record: bool = False,
    record_base_name: Optional[str] = None,
    logger=None,
) -> StatisticsCollector:
    """
    Run multiple games in parallel using process pool

    Args:
        num_games: Number of game instances to run
        num_threads: Number of worker processes
        width: Game world width
        height: Game world height
        ai_type: Type of AI to use ('heuristic' or 'neural')
        seed: Optional random seed for reproducibility
        show_progress: Whether to show progress updates
        model_path: Path to trained model weights (for neural AI)
        record: Whether to record game state and actions to individual NPZ files
        record_base_name: Base name for recording files (saved to ./data/{base_name}_{game_id}.npz)
        logger: Optional logger instance for dual output

    Returns:
        StatisticsCollector with all results
    """
    collector = StatisticsCollector()
    log = logger.info if logger else print

    # Create data directory if recording is enabled
    if record:
        if record_base_name is None:
            raise ValueError("record_base_name must be provided when record=True")
        os.makedirs("./data", exist_ok=True)
        log(
            f"Recording enabled. Files will be saved to ./data/{record_base_name}_<game_id>.npz"
        )

    # Load neural network model if needed
    ai_params = None
    if ai_type == "neural":
        import torch

        from asteroids.ai import NNAIParameters
        from asteroids.ai import validate_and_load_model

        # Default to nn_model.pth if no path specified
        if model_path is None:
            model_path = "nn_model.pth"

        params = NNAIParameters(device="cpu")
        log(f"loading model file {model_path}")
        validate_and_load_model(
            params.model,
            torch.load("nn_weights/" + model_path, map_location="cpu"),
            source_description=model_path,
        )
        params.model.eval()
        ai_params = params

    # Prepare arguments for each game
    game_args = [
        (game_id, width, height, ai_type, seed, ai_params, record_base_name)
        for game_id in range(num_games)
    ]

    log(f"Running {num_games} games on {num_threads} processes...")
    log(f"AI Type: {ai_type}")
    if seed is not None:
        log(f"Random Seed: {seed}")
    log("")

    # Use ProcessPoolExecutor for parallel execution (bypasses GIL)
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        # Submit all games
        futures = [executor.submit(run_single_game, args) for args in game_args]

        # Process results as they complete
        completed = 0
        start_time = time.time()

        for future in as_completed(futures):
            result = future.result()
            collector.add_result(result)
            completed += 1

            if show_progress and completed % max(1, num_games // 20) == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (num_games - completed) / rate if rate > 0 else 0

                # Calculate current average score
                current_avg = sum(r["score"] for r in collector.results) / len(
                    collector.results
                )

                log(
                    f"Progress: {completed}/{num_games} "
                    f"({100*completed/num_games:.0f}%) - "
                    f"Avg Score: {current_avg:.1f} - "
                    f"Rate: {rate:.1f} games/sec - "
                    f"ETA: {eta:.1f}s"
                )

    elapsed_total = time.time() - start_time
    log(f"\nCompleted {num_games} games in {elapsed_total:.1f} seconds")
    log(f"Average rate: {num_games/elapsed_total:.1f} games/second")

    return collector


def main():
    parser = argparse.ArgumentParser(
        description="Run headless Asteroids game simulations for AI benchmarking"
    )

    # Core parameters
    parser.add_argument(
        "-n",
        "--num-games",
        type=int,
        default=10,
        help="Number of game instances to run (default: 10)",
    )

    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=os.cpu_count() or 4,
        help=f"Number of worker processes (default: {os.cpu_count() or 4})",
    )

    parser.add_argument(
        "--ai-type",
        choices=["heuristic", "neural"],
        default="heuristic",
        help="Type of AI to use (default: heuristic)",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model weights file (for neural AI, default: nn_model.pth)",
    )

    # Game configuration
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Game world width (default: 1280)",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Game world height (default: 720)",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results (optional)",
    )

    # Output options
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )

    parser.add_argument(
        "--record",
        nargs="?",
        const="game_recordings",
        default=None,
        metavar="BASENAME",
        help="Record game state and actions to ./data/<basename>_<game_id>.npz files (default: game_recordings)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.num_games < 1:
        parser.error("Number of games must be at least 1")
    if args.threads < 1:
        parser.error("Number of threads must be at least 1")

    # Set up logging if recording is enabled
    logger = None
    if args.record is not None:
        logger = setup_headless_logging(args.record)

    # Run simulations
    collector = run_parallel_games(
        num_games=args.num_games,
        num_threads=args.threads,
        width=args.width,
        height=args.height,
        ai_type=args.ai_type,
        seed=args.seed,
        show_progress=not args.no_progress,
        model_path=args.model_path,
        record=args.record is not None,
        record_base_name=args.record,
        logger=logger,
    )

    # Print summary
    collector.print_summary(logger)


if __name__ == "__main__":
    main()
