"""
Genetic algorithm for optimizing AI parameters.

This script evolves AI parameters through generations of simulated games,
using selection, crossover, and mutation to find optimal configurations.
"""

import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from game_runner import run_single_game
from heuristic_ai_input import SmartAIInputParameters

# Parameter bounds for each AI parameter
# Format: {param_name: (min_value, max_value)}
PARAM_BOUNDS = {
    "evasion_max_distance": (300, 800),
    "max_speed": (50, 200),
    "evasion_lookahead_ticks": (10, 100),
    "shoot_angle_tolerance": (0.01, 0.2),
    "movement_angle_tolerance": (0.05, 0.3),
}


class Individual:
    """
    Represents one candidate solution (set of AI parameters).

    Attributes:
        params: SmartAIInputParameters instance
        fitness: Average score from evaluation games (None if not evaluated)
    """

    def __init__(self, params: SmartAIInputParameters):
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
        # TODO: Implement random parameter generation
        population = []
        for _ in range(self.population_size):
            params = self._random_params()
            individual = Individual(params)
            population.append(individual)
        return population

    def _random_params(self) -> SmartAIInputParameters:
        """Generate random AI parameters within bounds"""
        # TODO: Implement random parameter generation using PARAM_BOUNDS
        # For each parameter in PARAM_BOUNDS:
        #   - Get (min_val, max_val) from bounds
        #   - Generate random value: random.uniform(min_val, max_val)
        #   - For integer parameters (like evasion_lookahead_ticks), use random.randint
        # Return SmartAIInputParameters with generated values
        pass

    def evaluate_population(self, population: list[Individual]):
        """
        Evaluate fitness of all individuals in parallel.

        For each individual:
        - Run games_per_individual games with its parameters
        - Set fitness to average score

        Args:
            population: List of individuals to evaluate
        """
        # TODO: Implement parallel evaluation using ProcessPoolExecutor
        # Similar to run_parallel_games in main_headless, but with different params per individual
        #
        # For each individual in population:
        #   1. Create game_args list with games_per_individual games
        #   2. Each game should use individual.params as ai_params
        #   3. Submit all games to executor
        #   4. Collect results and calculate average score
        #   5. Set individual.fitness to average score
        pass

    def selection(self, population: list[Individual], k: int) -> list[Individual]:
        """
        Select k individuals using tournament selection.

        Args:
            population: Current population
            k: Number of individuals to select

        Returns:
            List of selected individuals
        """
        # TODO: Implement tournament selection
        # Tournament size: 3-5 individuals
        #
        # For i in range(k):
        #   1. Randomly select tournament_size individuals from population
        #   2. Choose the one with highest fitness
        #   3. Add to selected list
        # Return selected list
        pass

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
        # TODO: Implement blend crossover
        # For each parameter: offspring = alpha * p1 + (1-alpha) * p2
        # where alpha is random in range [0, 1]
        #
        # Example:
        #   alpha = random.random()
        #   offspring_evasion = alpha * parent1.params.evasion_max_distance +
        #                       (1-alpha) * parent2.params.evasion_max_distance
        #
        # Create new SmartAIInputParameters with blended values
        # Return new Individual with blended params
        pass

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
        # TODO: Implement Gaussian mutation
        # sigma = (max - min) * 0.1  # 10% of range
        #
        # For each parameter:
        #   if random.random() < self.mutation_rate:
        #     1. Get bounds from PARAM_BOUNDS
        #     2. Calculate sigma = (max - min) * 0.1
        #     3. Add Gaussian noise: param += random.gauss(0, sigma)
        #     4. Clamp to bounds: param = max(min_val, min(max_val, param))
        #
        # Create new SmartAIInputParameters with mutated values
        # Return new Individual
        pass

    def evolve(self) -> Individual:
        """
        Main evolution loop.

        For each generation:
        1. Evaluate population fitness
        2. Select parents
        3. Create offspring through crossover and mutation
        4. Replace population (keep elites)
        5. Track best individual

        Returns:
            Best individual found across all generations
        """
        print(f"Genetic Algorithm: Optimizing {self.ai_type} AI Parameters")
        print(f"Population: {self.population_size}, Generations: {self.generations}")
        print(f"Games per individual: {self.games_per_individual}")
        print()

        # Initialize population
        self.population = self.initialize_population()

        for gen in range(self.generations):
            self.generation = gen + 1

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

            # Print progress
            self._print_generation_stats()

            # Last generation - no need to create offspring
            if gen == self.generations - 1:
                break

            # Create next generation
            next_population = []

            # Elitism: keep best individuals
            next_population.extend(self.population[: self.elite_size])

            # Fill rest with offspring
            while len(next_population) < self.population_size:
                # Selection
                parents = self.selection(self.population, 2)

                # Crossover
                if random.random() < self.crossover_rate:
                    offspring = self.crossover(parents[0], parents[1])
                else:
                    offspring = parents[0]  # Clone first parent

                # Mutation
                if random.random() < self.mutation_rate:
                    offspring = self.mutate(offspring)

                next_population.append(offspring)

            self.population = next_population

        return self.best_ever

    def _print_generation_stats(self):
        """Print statistics for current generation"""
        fitnesses = [ind.fitness for ind in self.population]
        best = max(fitnesses)
        avg = sum(fitnesses) / len(fitnesses)
        worst = min(fitnesses)

        # Calculate diversity (standard deviation of parameters)
        diversity = self._calculate_diversity()

        print(
            f"Generation {self.generation}/{self.generations} - "
            f"Best: {best:.1f} - Avg: {avg:.1f} - Worst: {worst:.1f} - "
            f"Diversity: {diversity:.2f}"
        )

    def _calculate_diversity(self) -> float:
        """
        Calculate population diversity.

        Returns:
            Diversity metric (0-1, higher = more diverse)
        """
        # TODO: Implement diversity calculation
        # Could use std dev of each parameter, normalized
        #
        # For each parameter:
        #   1. Collect values across all individuals
        #   2. Calculate standard deviation
        #   3. Normalize by parameter range: std_dev / (max - min)
        #   4. Average normalized std_devs across all parameters
        #
        # Return average normalized diversity
        return 0.0


def print_best_parameters(
    individual: Individual, baseline_fitness: Optional[float] = None
):
    """
    Print the best parameters found.

    Args:
        individual: Best individual
        baseline_fitness: Optional baseline fitness for comparison
    """
    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Best fitness: {individual.fitness:.1f}")
    print()
    print("Best parameters:")
    print(f"  evasion_max_distance:     {individual.params.evasion_max_distance}")
    print(f"  max_speed:                {individual.params.max_speed}")
    print(f"  evasion_lookahead_ticks:  {individual.params.evasion_lookahead_ticks}")
    print(f"  shoot_angle_tolerance:    {individual.params.shoot_angle_tolerance:.3f}")
    print(
        f"  movement_angle_tolerance: {individual.params.movement_angle_tolerance:.3f}"
    )

    if baseline_fitness is not None:
        improvement = ((individual.fitness - baseline_fitness) / baseline_fitness) * 100
        print()
        print(f"Baseline fitness: {baseline_fitness:.1f}")
        print(f"Improvement: {improvement:+.1f}%")
    print("=" * 60)


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
        type=str,
        default="smart",
        help="AI type to optimize (default: smart)",
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

    print(f"\nTotal evolution time: {elapsed:.1f} seconds")
    print_best_parameters(best, args.baseline_fitness)


if __name__ == "__main__":
    main()
