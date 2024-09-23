
# Import classes and functions
from functions import Problem
from individual import Individual, Population, Codec

# For binary representation
from operators import roulette_wheel, single_point_crossover, binary_mutation

# For real representation
from operators import binary_tournament, simulated_binary_crossover, parameter_based_mutation

# TODO: What should I do with individuals outside the constraints?
#! TODO: Binary tournament doesn't have problem with this, but roulette wheel yes.
# TODO: +-Infinite fitness, but in selection, should I dispose them?

# TODO: Some genotypes allows phenotypes outside constraints (Which is correct).
# TODO: Mutation should skip sign bits (Binary Encoding)?

# Define parameters
population_size = 100
number_of_generations = 100

# Define representation
representation = "real"

# Define precision (only for binary)
precision = 4

# Define operators
if representation == "binary":
    selection = roulette_wheel
    crossover = single_point_crossover
    mutation = binary_mutation

else:
    selection = binary_tournament
    crossover = simulated_binary_crossover
    mutation = parameter_based_mutation

# Define problem
expression = "100*(x**2 - y**2) + (1 - x)**2"
variable_names = ["x", "y"]
num_variables = len(variable_names)

# Define constraints
intervals = [-2.048, 2.048]

# Define kind of optimization
minimize = False

def evaluate(population: Population, problem: Problem) -> None:
    """
    Parameters
    ----------
    population: Population
        Population object with individuals to be evaluated.
    problem: Problem
        Problem object to evaluate an individual.
    
    Returns
    -------
    None
    """

    for individual in population.get_population():
        # Get phenotype
        phenotype = dict(zip(variable_names, individual.phenotype))

        # Evaluate phenotype
        fitness = problem.evaluate(**phenotype)

        # Assign fitness
        individual.fitness = fitness


def step(population: Population, codec: Codec, rates: list[float]) -> list[Individual]:
    """
    Parameters
    ----------
    population: Population
        Population object with individuals.
    codec: Codec
        Codec object to encode and decode individuals.
    rates: list[float]
        List with crossover and mutation rates.
    
    Returns
    -------
    list[Individuals]
        Individuals of the new population
    """

    # Unpack rates
    crossover_rate, mutation_rate = rates

    # Initialize list for new population
    new_population = []

    while len(new_population) < population_size:

        # Select parents
        parents = selection(population.get_population(), minimize)

        # Crossover
        offspring = crossover(parents, crossover_rate)

        # Mutation
        offspring = mutation(offspring, mutation_rate)

        # Decode individuals
        for child in offspring:
            phenotype = codec.decode(child.genome)
            child.phenotype = phenotype
        
        # Append to new population
        new_population.extend(offspring)

    return new_population


def main():

    # Initialize problem
    constraints  = [tuple(intervals) for _ in range(num_variables)]
    problem = Problem(expression, num_variables, constraints, minimize)

    # Initialize a codec to handle genome encodings and decodings
    codec = Codec(kind=representation, precision=precision, intervals=intervals)

    #  Initialize population
    population  = Population(population_size)
    population.initialize(num_variables, intervals, codec)

    # Evaluate initial population
    evaluate(population, problem)

    # Define probabilities
    crossover_rate = 0.9
    mutation_rate = 1 / (num_variables * (codec.bit_length + 1))

    genetic_rates = [crossover_rate, mutation_rate]

    for gen in range(number_of_generations):

        # Do an evolutionary step
        new_population = step(population, codec, genetic_rates)

        # Replace population (1 to 1)
        population.update_population(new_population)

        # Evaluate population
        evaluate(population, problem)

        # Save statistics
        fitness_values = population.get_fitness()
        if minimize:
            print(min(fitness_values))
        else:
            print(max(fitness_values))
        

if __name__ == "__main__":
    main()