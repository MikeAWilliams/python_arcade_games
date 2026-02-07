"""
Genetic algorithm for optimizing AI parameters.

This script evolves AI parameters through generations of simulated games,
using selection, crossover, and mutation to find optimal configurations.
"""

import argparse
import logging
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

# Add parent directory to path so we can import asteroids package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asteroids.core.game_runner import run_single_game


def setup_genetic_logging():
    """
    Set up dual logging (console + file) for genetic algorithm.
    Log file saved to root directory as genetic.log (overwritten each run).
    Returns: logger instance
    """
    log_file = "genetic.log"

    # Configure logger
    logger = logging.getLogger("genetic")
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


class Individual:
    """
    Represents one candidate solution (set of AI parameters).

    Attributes:
        params: AI parameter object (type depends on AI type)
        fitness: Average score from evaluation games (None if not evaluated)
    """

    def __init__(self, params):
        self.params = params
        self.fitness: Optional[float] = None

    def __repr__(self):
        return f"Individual(fitness={self.fitness})"


class GeneticAlgorithm:
    """
    Genetic algorithm for evolving AI parameters.

    Uses:
    - Tournament selection
    - Blend crossover (average parameters from two parents)
    - Gaussian mutation
    - Elitism (keep best individuals)
    """

    def __init__(
        self,
        population_size: int,
        generations: int,
        mutation_rate: float,
        crossover_rate: float,
        elite_size: int,
        games_per_individual: int,
        num_threads: int,
        ai_type: str,
        width: int,
        height: int,
        seed: Optional[int] = None,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.games_per_individual = games_per_individual
        self.num_threads = num_threads
        self.ai_type = ai_type
        self.width = width
        self.height = height
        self.seed = seed

        self.generation = 0
        self.population: list[Individual] = []
        self.best_ever: Optional[Individual] = None

    def initialize_population(self) -> list[Individual]:
        """
        Create initial population with random parameters within bounds.

        Returns:
            List of Individual instances with random parameters
        """
        population = []
        for _ in range(self.population_size):
            params = self._random_params()
            individual = Individual(params)
            population.append(individual)
        return population

    def _random_params(self):
        """Generate random AI parameters within bounds"""
        if self.ai_type == "heuristic":
            from asteroids.ai.heuristic import heuristic_ai_random_params

            return heuristic_ai_random_params()
        else:
            raise ValueError(
                f"No parameter generation function for AI type: {self.ai_type}"
            )

    def evaluate_population(self, population: list[Individual]):
        """
        Evaluate fitness of all individuals in parallel.

        For each individual:
        - Run games_per_individual games with its parameters
        - Set fitness to average score

        Args:
            population: List of individuals to evaluate
        """
        # Prepare all game arguments
        all_game_args = []
        individual_game_counts = []

        game_id = 0
        for individual in population:
            # Create games for this individual
            for _ in range(self.games_per_individual):
                args = (
                    game_id,
                    self.width,
                    self.height,
                    self.ai_type,
                    self.seed,
                    individual.params,
                )
                all_game_args.append(args)
                game_id += 1
            individual_game_counts.append(self.games_per_individual)

        # Run all games in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(run_single_game, args) for args in all_game_args]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # Sort results by game_id to match with individuals
        results.sort(key=lambda r: r["game_id"])

        # Calculate fitness for each individual
        result_idx = 0
        for i, individual in enumerate(population):
            # Get results for this individual's games
            individual_results = results[
                result_idx : result_idx + individual_game_counts[i]
            ]
            result_idx += individual_game_counts[i]

            # Calculate average score as fitness
            scores = [r["score"] for r in individual_results]
            individual.fitness = sum(scores) / len(scores)

    def selection(self, population: list[Individual], k: int) -> list[Individual]:
        """
        Select k individuals using tournament selection.

        Args:
            population: Current population
            k: Number of individuals to select

        Returns:
            List of selected individuals
        """
        tournament_size = 3
        selected = []

        for _ in range(k):
            # Randomly select tournament_size individuals
            tournament = random.sample(population, tournament_size)

            # Choose the one with highest fitness
            winner = max(tournament, key=lambda ind: ind.fitness)

            selected.append(winner)

        return selected

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Create offspring by blending two parents' parameters.

        Uses blend crossover: for each parameter, offspring value is
        randomly chosen from range [parent1_val, parent2_val] or
        averaged with some randomness.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            New individual (offspring)
        """
        if self.ai_type == "heuristic":
            from asteroids.ai.heuristic import heuristic_ai_crossover

            offspring_params = heuristic_ai_crossover(parent1.params, parent2.params)
            return Individual(offspring_params)
        else:
            raise ValueError(f"No crossover function for AI type: {self.ai_type}")

    def mutate(self, individual: Individual) -> Individual:
        """
        Mutate individual's parameters with Gaussian noise.

        For each parameter, with probability mutation_rate:
        - Add Gaussian noise: param += N(0, sigma)
        - Clamp to parameter bounds

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual (new instance)
        """
        if self.ai_type == "heuristic":
            from asteroids.ai.heuristic import heuristic_ai_mutate

            mutated_params = heuristic_ai_mutate(individual.params, self.mutation_rate)
            return Individual(mutated_params)
        else:
            raise ValueError(f"No mutation function for AI type: {self.ai_type}")

    def evolve(self) -> Individual:
        """
        Main evolution loop.

        Returns:
            Best individual found across all generations
        """
        # Set up logging
        self.logger = setup_genetic_logging()

        self.logger.info(f"Genetic Algorithm: Optimizing {self.ai_type} AI Parameters")
        self.logger.info(
            f"Population: {self.population_size}, Generations: {self.generations}"
        )
        self.logger.info(f"Games per individual: {self.games_per_individual}")
        self.logger.info("")

        # Initialize population
        self.population = self.initialize_population()

        start_time = time.time()
        generation_times = []

        for gen in range(self.generations):
            self.generation = gen + 1
            gen_start = time.time()

            # Evaluate fitness
            self.evaluate_population(self.population)

            # Sort by fitness (descending)
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)

            # Track best ever
            if (
                self.best_ever is None
                or self.population[0].fitness > self.best_ever.fitness
            ):
                self.best_ever = self.population[0]

            # Timing calculations
            gen_duration = time.time() - gen_start
            generation_times.append(gen_duration)

            elapsed = time.time() - start_time
            avg_gen_time = sum(generation_times) / len(generation_times)
            est_total = avg_gen_time * self.generations
            remaining = est_total - elapsed

            # Print progress
            self._print_generation_stats(elapsed, est_total, remaining)

            # Last generation - no need to create offspring
            if gen == self.generations - 1:
                break

            # Create next generation
            next_population = []

            # Elitism
            next_population.extend(self.population[: self.elite_size])

            # Fill rest with offspring
            while len(next_population) < self.population_size:
                parents = self.selection(self.population, 2)

                if random.random() < self.crossover_rate:
                    offspring = self.crossover(parents[0], parents[1])
                else:
                    offspring = parents[0]

                if random.random() < self.mutation_rate:
                    offspring = self.mutate(offspring)

                next_population.append(offspring)

            self.population = next_population

        return self.best_ever

    def _print_generation_stats(self, elapsed, total, remaining):
        """Print statistics for current generation"""
        fitnesses = [ind.fitness for ind in self.population]
        best = max(fitnesses)
        avg = sum(fitnesses) / len(fitnesses)
        worst = min(fitnesses)

        # Calculate diversity (standard deviation of parameters)
        diversity = self._calculate_diversity()

        self.logger.info(
            f"Generation {self.generation}/{self.generations} - "
            f"Best: {best:.1f} - Avg: {avg:.1f} - Worst: {worst:.1f} - "
            f"Diversity: {diversity:.2f} - "
            f"Time elapsed: {elapsed:.1f}s - "
            f"Estimated total: {total:.1f}s - "
            f"Estimated remaining: {max(0, remaining):.1f}s"
        )

    def _calculate_diversity(self) -> float:
        """
        Calculate population diversity.

        Returns:
            Diversity metric (0-1, higher = more diverse)
        """
        if self.ai_type == "heuristic":
            from asteroids.ai.heuristic import heuristic_ai_calculate_diversity

            params_list = [ind.params for ind in self.population]
            return heuristic_ai_calculate_diversity(params_list)
        else:
            # Return 0 for unknown AI types
            return 0.0


def print_best_parameters(
    individual: Individual,
    ai_type: str,
    baseline_fitness: Optional[float] = None,
    logger=None,
):
    """
    Print the best parameters found.

    Args:
        individual: Best individual
        ai_type: AI type being optimized
        baseline_fitness: Optional baseline fitness for comparison
        logger: Logger instance to use (if None, uses print)
    """
    log = logger.info if logger else print

    log("\n" + "=" * 60)
    log("EVOLUTION COMPLETE")
    log("=" * 60)
    log(f"Best fitness: {individual.fitness:.1f}")
    log("")
    log("Best parameters:")

    # Print parameters based on AI type
    if ai_type == "heuristic":
        params = individual.params
        log(f"  evasion_max_distance:     {params.evasion_max_distance}")
        log(f"  max_speed:                {params.max_speed}")
        log(f"  evasion_lookahead_ticks:  {params.evasion_lookahead_ticks}")
        log(f"  shoot_angle_tolerance:    {params.shoot_angle_tolerance:.3f}")
        log(f"  movement_angle_tolerance: {params.movement_angle_tolerance:.3f}")
    else:
        # Generic fallback - print all attributes
        for attr, value in vars(individual.params).items():
            log(f"  {attr}: {value}")

    if baseline_fitness is not None:
        improvement = ((individual.fitness - baseline_fitness) / baseline_fitness) * 100
        log("")
        log(f"Baseline fitness: {baseline_fitness:.1f}")
        log(f"Improvement: {improvement:+.1f}%")
    log("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Genetic algorithm for optimizing AI parameters"
    )

    # GA parameters
    parser.add_argument(
        "--population",
        type=int,
        default=50,
        help="Population size (default: 50)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of generations (default: 20)",
    )
    parser.add_argument(
        "--games-per-individual",
        type=int,
        default=10,
        help="Number of games to evaluate each individual (default: 10)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.2,
        help="Mutation probability (default: 0.2)",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.7,
        help="Crossover probability (default: 0.7)",
    )
    parser.add_argument(
        "--elite-size",
        type=int,
        default=5,
        help="Number of elites to keep each generation (default: 5)",
    )

    # AI and game parameters
    parser.add_argument(
        "--ai-type",
        choices=["heuristic"],
        default="heuristic",
        help="AI type to optimize (default: heuristic)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Game width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Game height (default: 720)",
    )

    # Execution parameters
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=os.cpu_count() or 4,
        help=f"Number of parallel processes (default: {os.cpu_count() or 4})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )

    # Baseline comparison
    parser.add_argument(
        "--baseline-fitness",
        type=float,
        default=None,
        help="Baseline fitness for comparison (optional)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.population < args.elite_size:
        parser.error("Population size must be >= elite size")
    if args.mutation_rate < 0 or args.mutation_rate > 1:
        parser.error("Mutation rate must be in [0, 1]")
    if args.crossover_rate < 0 or args.crossover_rate > 1:
        parser.error("Crossover rate must be in [0, 1]")

    # Create and run GA
    ga = GeneticAlgorithm(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elite_size=args.elite_size,
        games_per_individual=args.games_per_individual,
        num_threads=args.threads,
        ai_type=args.ai_type,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    start_time = time.time()
    best = ga.evolve()
    elapsed = time.time() - start_time

    ga.logger.info(f"\nTotal evolution time: {elapsed:.1f} seconds")
    print_best_parameters(best, args.ai_type, args.baseline_fitness, ga.logger)


if __name__ == "__main__":
    main()
