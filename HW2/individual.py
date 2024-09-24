import numpy as np
from codec import Codec

class Individual():

    def __init__(self, genome: list[float] |  str) -> None:
        
        self.genome = genome
        self.phenotype: list[float] | None = None

        self.fitness: float|None = None

    def set_phenotype(self, phenotype: list[float]) -> None:
        self.phenotype = phenotype

    def __repr__(self):
        return f"Individual(phenotype={self.phenotype}, fitness={self.fitness})"


class Population():

    def __init__(self, pop_size: int) -> None:
    
        self.pop_size = pop_size
        self.population: list[Individual] = []
    

    def initialize(self, num_variables: int, intervals: tuple[float, float], codec: Codec) -> None:
        
        # Random initialization
        interval = intervals[1] - intervals[0]
        for _ in range(self.pop_size):

            # Make a list of random numbers
            random_list: list[float] = (np.random.rand(num_variables) * interval + intervals[0]).tolist()

            # Encode the random list to get the genome
            genome: list[float] | str = codec.encode(random_list)

            # Create an individual
            individual = Individual(genome)

            # Set its phenotype
            individual.set_phenotype(random_list)

            # Add it to the population
            self.population.append(individual)

    def get_population(self) -> list[Individual]:
        return self.population

    def update_population(self, new_population: list[Individual]) -> None:
        self.population = new_population

    def get_fitness(self) -> list[float]:
        fitness_values: list[float] = []
        for individual in self.population:
            fitness_values.append(individual.fitness)
        
        return fitness_values
