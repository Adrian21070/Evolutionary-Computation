
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection

from functions import Problem
from individual import Population

class Graphics():

    def __init__(self, minimize: bool) -> None:

        # Set optimization problem
        self.minimize = minimize

        # Initialize data structures to store fitness values
        self.fitness_history: list[float] = []

        self.best_solutions: list[list] = []


    def extract_population_data(self, population: Population):

        # Get individuals
        individuals = population.get_population()

        # Solution set
        phenotypes = np.array([individual.phenotype for individual in individuals])

        # Objective value
        fitness = np.array([individual.fitness for individual in individuals])

        return phenotypes, fitness

    def create_3d_plot(self, population: Population, problem: Problem, intervals: list[float]):

        # Make 3D surface
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})

        ## Make points
        x = np.linspace(intervals[0], intervals[1], 100)
        X,Y = np.meshgrid(x,x)
        
        ## Evaluate Points
        Z = problem.evaluate(x=X, y=Y)

        ## Plot surface
        ax.plot_surface(X, Y, Z, alpha=0.5)

        # Plot solutions
        
        ## Get data
        variable_space, objective_space = self.extract_population_data(population)

        ## Scatter plot
        points = ax.scatter(variable_space[:,0], variable_space[:,1], objective_space, color="r")

        # Render image
        plt.pause(0.1)

        # Store data
        self.ax = ax
        self.points = points
    
    def update_3d_plot(self, population: Population):

        # Get data
        variable_space, objective_space = self.extract_population_data(population)

        # Delete old points
        self.points.remove()

        # Scatter new points
        points = self.ax.scatter(variable_space[:,0], variable_space[:,1], objective_space, color="r")

        # Render
        plt.pause(0.1)

        # Update values
        self.points = points

    def plot_fitness(self):
        
        fig, ax = plt.subplots(1,1)

        x = range(len(self.fitness_history))

        ax.plot(x, self.fitness_history)

        plt.show()


    def save_best_individual(self, population: Population) -> tuple[list[float], float]:
        
        # Get individuals
        individuals = population.get_population()

        # Get fitness
        fitness_values = population.get_fitness()

        # Get min/max fitness index
        value = min(fitness_values) if self.minimize else max(fitness_values)        
        index = fitness_values.index(value)

        # Get best individual
        best_individual = individuals[index]

        fitness = round(best_individual.fitness, 4)
        phenotype = [round(value, 4) for value in best_individual.phenotype]

        # Store values
        self.fitness_history.append(fitness)
        self.best_solutions.append(phenotype)

        return phenotype, fitness
    