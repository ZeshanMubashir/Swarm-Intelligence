"""
Artificial Bee Colony (ABC) Algorithm Implementation

The ABC algorithm is a swarm intelligence-based optimization algorithm inspired by
the foraging behavior of honey bees. It consists of three types of bees:
- Employed bees: Search for food sources (solutions)
- Onlooker bees: Select food sources based on probability
- Scout bees: Discover new food sources when existing ones are exhausted
"""

import random
import math


class BeeColony:
    """Artificial Bee Colony optimization algorithm."""

    def __init__(
        self,
        objective_function,
        dimensions,
        bounds,
        colony_size=30,
        max_iterations=100,
        limit=50,
    ):
        """
        Initialize the Bee Colony.

        Args:
            objective_function: The function to minimize
            dimensions: Number of dimensions in the search space
            bounds: Tuple of (min_val, max_val) for each dimension
            colony_size: Number of food sources (half will be employed bees)
            max_iterations: Maximum number of iterations
            limit: Number of trials before a food source is abandoned
        """
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.bounds = bounds
        self.colony_size = colony_size
        self.max_iterations = max_iterations
        self.limit = limit

        # Number of food sources equals number of employed bees
        self.num_food_sources = colony_size // 2

        # Initialize food sources and their fitness values
        self.food_sources = []
        self.fitness_values = []
        self.trial_counters = []

        # Best solution found
        self.best_solution = None
        self.best_fitness = float("inf")

    def _initialize_food_sources(self):
        """Initialize random food sources within bounds."""
        self.food_sources = []
        self.fitness_values = []
        self.trial_counters = []

        for _ in range(self.num_food_sources):
            food_source = [
                random.uniform(self.bounds[0], self.bounds[1])
                for _ in range(self.dimensions)
            ]
            self.food_sources.append(food_source)
            fitness = self._calculate_fitness(food_source)
            self.fitness_values.append(fitness)
            self.trial_counters.append(0)

            # Update best solution
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = food_source.copy()

    def _calculate_fitness(self, solution):
        """Calculate fitness value for a solution."""
        return self.objective_function(solution)

    def _employed_bee_phase(self):
        """Employed bees search for new food sources."""
        for i in range(self.num_food_sources):
            # Generate new candidate solution
            new_solution = self._generate_new_solution(i)

            # Evaluate new solution
            new_fitness = self._calculate_fitness(new_solution)

            # Greedy selection
            if new_fitness < self.fitness_values[i]:
                self.food_sources[i] = new_solution
                self.fitness_values[i] = new_fitness
                self.trial_counters[i] = 0

                # Update best solution
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_solution.copy()
            else:
                self.trial_counters[i] += 1

    def _onlooker_bee_phase(self):
        """Onlooker bees select food sources based on probability."""
        # Calculate selection probabilities
        transformed = [1.0 / (1.0 + f) for f in self.fitness_values]
        fitness_sum = sum(transformed)
        probabilities = [t / fitness_sum for t in transformed]

        for _ in range(self.num_food_sources):
            # Select food source based on probability
            selected_index = self._roulette_wheel_selection(probabilities)

            # Generate new candidate solution
            new_solution = self._generate_new_solution(selected_index)

            # Evaluate new solution
            new_fitness = self._calculate_fitness(new_solution)

            # Greedy selection
            if new_fitness < self.fitness_values[selected_index]:
                self.food_sources[selected_index] = new_solution
                self.fitness_values[selected_index] = new_fitness
                self.trial_counters[selected_index] = 0

                # Update best solution
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_solution.copy()
            else:
                self.trial_counters[selected_index] += 1

    def _scout_bee_phase(self):
        """Scout bees discover new food sources."""
        for i in range(self.num_food_sources):
            if self.trial_counters[i] >= self.limit:
                # Abandon food source and discover new one
                new_source = [
                    random.uniform(self.bounds[0], self.bounds[1])
                    for _ in range(self.dimensions)
                ]
                self.food_sources[i] = new_source
                self.fitness_values[i] = self._calculate_fitness(new_source)
                self.trial_counters[i] = 0

                # Update best solution
                if self.fitness_values[i] < self.best_fitness:
                    self.best_fitness = self.fitness_values[i]
                    self.best_solution = self.food_sources[i].copy()

    def _generate_new_solution(self, index):
        """Generate a new candidate solution near the given food source."""
        new_solution = self.food_sources[index].copy()

        # Select random dimension to modify
        dim = random.randint(0, self.dimensions - 1)

        # Select random partner (different from current)
        candidates = [i for i in range(self.num_food_sources) if i != index]
        partner = random.choice(candidates)

        # Generate new value
        phi = random.uniform(-1, 1)
        new_solution[dim] = (
            self.food_sources[index][dim]
            + phi * (self.food_sources[index][dim] - self.food_sources[partner][dim])
        )

        # Ensure within bounds
        new_solution[dim] = max(self.bounds[0], min(self.bounds[1], new_solution[dim]))

        return new_solution

    def _roulette_wheel_selection(self, probabilities):
        """Select an index based on roulette wheel selection."""
        r = random.random()
        cumulative = 0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return i
        return len(probabilities) - 1

    def optimize(self):
        """Run the optimization algorithm."""
        self._initialize_food_sources()

        for iteration in range(self.max_iterations):
            self._employed_bee_phase()
            self._onlooker_bee_phase()
            self._scout_bee_phase()

        return self.best_solution, self.best_fitness


def sphere_function(x):
    """Sphere function - a simple test function for optimization."""
    return sum(xi**2 for xi in x)


def rastrigin_function(x):
    """Rastrigin function - a more complex test function with many local minima."""
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def main():
    """Demonstrate the Artificial Bee Colony algorithm."""
    print("=" * 60)
    print("Artificial Bee Colony (ABC) Algorithm Demonstration")
    print("=" * 60)

    # Test with Sphere function
    print("\n1. Optimizing Sphere Function (global minimum at origin = 0)")
    print("-" * 60)

    abc = BeeColony(
        objective_function=sphere_function,
        dimensions=5,
        bounds=(-10, 10),
        colony_size=30,
        max_iterations=100,
        limit=50,
    )

    best_solution, best_fitness = abc.optimize()

    print(f"Best solution found: {[round(x, 6) for x in best_solution]}")
    print(f"Best fitness value: {best_fitness:.6f}")

    # Test with Rastrigin function
    print("\n2. Optimizing Rastrigin Function (global minimum at origin = 0)")
    print("-" * 60)

    abc = BeeColony(
        objective_function=rastrigin_function,
        dimensions=5,
        bounds=(-5.12, 5.12),
        colony_size=50,
        max_iterations=200,
        limit=100,
    )

    best_solution, best_fitness = abc.optimize()

    print(f"Best solution found: {[round(x, 6) for x in best_solution]}")
    print(f"Best fitness value: {best_fitness:.6f}")

    print("\n" + "=" * 60)
    print("Optimization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
