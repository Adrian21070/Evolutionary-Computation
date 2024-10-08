
import random
import numpy as np
from individual import Individual

from typing import Union, Optional

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

    # Allocate memory for parents
    parents = []
    num_parents = 2

    # Get fitness
    fitness_values = [individual.fitness for individual in population]

    # Adjust fitness values if minimization problem.
    if minimize:
        max_fitness = max(fitness_values)
        adjusted_fitness = [max_fitness - fitness for fitness in fitness_values]
    else:
        adjusted_fitness = fitness_values
    
    # Wheel table
    cumulated_fitness = np.sum(adjusted_fitness)
    probability_fitness = [(fitness_value / cumulated_fitness) for fitness_value in adjusted_fitness]
    cumulated_probability = [(proba + np.sum(probability_fitness[:i]))  for i, proba  in enumerate(probability_fitness)]
    
    for _ in range(num_parents):
        r = random.random()
        
        for i, proba in enumerate(cumulated_probability):
            if r <= proba:
                parents.append(population[i])
                break

    return parents

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
    if random.random() <= crossover_rate:
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

    random_position = random.randint(1, len(genome1)-1)

    # Convert to Individual
    if  random.random() <= crossover_rate:
        offspring1 = Individual(genome1[:random_position] + genome2[random_position:])
        offspring2 = Individual(genome2[:random_position] + genome1[random_position:])
        return [offspring1, offspring2]
    else:
        return parents


# ----MUTATION OPERATORS----

def parameter_based_mutation(offspring: list[Individual], mutation_rate: float,
                            intervals: list[float], nm: int = 20) -> list[Individual]:
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

        for i, value in enumerate(genome):
            
            # Random number to perform or not mutation.
            if random.random() > mutation_rate:
                continue

            # Perform mutation
            ## Get uniform value
            u = random.random()

            # Compute deltas
            delta_max = intervals[1] - intervals[0]
            delta = min((value - intervals[0]),(intervals[1] - value)) / delta_max

            # Compute delta q
            if u <= 0.5:
                delta_q = (2*u + (1-2*u)*((1-delta)**(nm + 1)))**(1/(nm+1)) - 1
            else:
                delta_q = 1 - (2*(1-u) + 2*(u-0.5)*((1-delta)**(nm+1)))**(1/(nm+1))
                
            # Mutate allele
            genome[i] = genome[i] + delta_q * delta_max
            
            # Done with this individual.
            break
        
    return offspring

def binary_mutation(offspring: list[Individual], mutation_rate: float,
                    intervals: Optional[list[float]] = None) -> list[Individual]:
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

    for individual in offspring:
        # Extract genome
        genome: list[str] = list(individual.genome)

        for index in range(len(genome)):

            if random.random() > mutation_rate:
                continue
            
            genome[index] = "0" if genome[index] == "1" else "1"
            break

        # Join the genome list.
        genome_str: str = "".join(genome)

        # Update the genome
        individual.genome = genome_str

    return offspring
