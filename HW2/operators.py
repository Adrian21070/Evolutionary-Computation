
import random
import numpy as np

from individual import Individual

# ----SELECTION OPERATORS----

def binary_tournament(population: list[Individual], minimize: bool) -> list[Individual]:
    """
    Parameters
    ----------
    population: list[Individual]
        List with individual objects.
    minimize: bool
        Bool to define minimization or maximization problem.
    
    Returns
    -------
    list[Individual]
        Selected parents.
    """
    
    # Binary tournament
    k = 2
    
    # Allocate memory for parents
    parents = []
    num_parents = 2

    # Define comparison sign
    sign = -1 if minimize else 1

    for _ in range(num_parents):

        # Get k random individuals
        selected_individuals = random.choices(population, k=k)

        # Get fitness values 
        # (Multiply by sign to adjust for minimization or maximization)
        fitness = [sign*individual.fitness for individual in selected_individuals]

        # Compare fitness
        if fitness[0] >= fitness[1]:
            best_individual = selected_individuals[0]
        else:
            best_individual = selected_individuals[1]
        
        parents.append(best_individual)

    return parents

def roulette_wheel(population: list[Individual], minimize: bool) -> list[Individual]:
    """
    Parameters
    ----------
    population: list[Individual]
        List with individual objects.
    minimize: bool
        Bool to define minimization or maximization problem.
    
    Returns
    -------
    list[Individual]
        Selected parents.
    """

    # Get fitness
    fitness_values = [individual.fitness for individual in population]

    # Adjust fitness values if minimization problem.
    if minimize:
        max_fitness = max(fitness_values)
        adjusted_fitness = [max_fitness - fitness for fitness in fitness_values]
    else:
        adjusted_fitness = fitness_values

    # TODO: Complete function...
    pass


# ----CROSSOVER OPERATORS----

def simulated_binary_crossover(parents: list[Individual], crossover_rate: float, nc: int = 20) -> list[Individual]:
    """
    Parameters
    ----------
    parents: list[Individual]
        List with parents (individual) objects.
    crossover_rate: float
        Probability of performing crossover
    nc: int
        Eta parameter to adjust perturbation distribution.
    
    Returns
    -------
    list[Individual]
        List with offspring.
    """

    # Unpack parents genomes
    parent1, parent2 = parents
    genome1, genome2 = parent1.genome, parent2.genome

    # Convert genomes list[floats] to numpy arrays
    genome1 = np.array(genome1)
    genome2 = np.array(genome2)

    # Generate n uniform random numbers [0,1]
    u = np.random.rand(len(genome1))

    # Compute beta when u[i] <= 0.5
    lower_beta = (2*u) ** (1 / (nc + 1))

    # Compute beta when u[i] > 0.5
    upper_beta = (1 / (2*(1-u))) ** (1 / (nc + 1))

    # Use numpy vectorized operations to allocate beta values
    beta = np.where(u <= 0.5, lower_beta, upper_beta)

    # Perform SBX
    if random.random() < crossover_rate:
        # Perform Crossover
        new_genome1 = 0.5 * ((1 + beta) * genome1 + (1 - beta) * genome2) 
        new_genome2 = 0.5 * ((1 - beta) * genome1 + (1 + beta) * genome2) 
    else:
        # No crossover
        new_genome1 = genome1
        new_genome2 = genome2

    # Convert to list
    new_genome1, new_genome2 = new_genome1.tolist(), new_genome2.tolist()

    # Convert to Individual
    offspring1 = Individual(new_genome1)
    offspring2 = Individual(new_genome2)

    return [offspring1, offspring2]

def single_point_crossover(parents: list[Individual], crossover_rate: float) -> list[Individual]:
    """
    Parameters
    ----------
    parents: list[Individual]
        List with parents (individual) objects.
    crossover_rate: float
        Probability of performing crossover
    
    Returns
    -------
    list[Individual]
        List with offspring.
    """
    
    # Unpack parents genomes
    parent1, parent2 = parents
    genome1, genome2 = parent1.genome, parent2.genome

    # Genomes are strings...
    # TODO: Implement


    # Convert to Individual
    offspring1 = Individual(new_genome1)
    offspring2 = Individual(new_genome2)

    return [offspring1, offspring2]


# ----MUTATION OPERATORS----

def parameter_based_mutation(offspring: list[Individual], mutation_rate: float, nm: int = 20) -> list[Individual]:
    """
    Parameters
    ----------
    offspring: list[Individual]
        List with offspring (individual) objects.
    mutation_rate: float
        Probability of performing mutation
    nm: int
        Eta parameter to adjust perturbation distribution.
    
    Returns
    -------
    list[Individual]
        List with mutated offspring.
    """

    for individual in offspring:

        # Extract genome
        genome: list[float] = individual.genome

        # Perform mutation
        pass

    return offspring

def binary_mutation(offspring: list[Individual], mutation_rate: float) -> list[Individual]:
    """
    Parameters
    ----------
    offspring: list[Individual]
        List with offspring (individual) objects.
    mutation_rate: float
        Probability of performing mutation

    Returns
    -------
    list[Individual]
        List with mutated offspring.
    """

    pass