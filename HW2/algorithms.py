
from abc import ABC, abstractmethod

import numpy as np


from codec import Codec
from graphics import Graphics
from functions import Problem
from individual import Population, Individual


class GeneticAlgorithm():

    def __init__(self, representation:str, selection: function,
                  crossover: function, mutation: function):
        
        
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.representation = representation

    def load_problem(self, expression: str, variables: list[str],
                      intervals: list[float], minimize: bool, precision: int=4):

        # Allocate variables
        self.minimize = minimize
        self.expression = expression

        self.variables = variables
        self.num_variables = len(variables)
        
        self.intervals = intervals

        # Get constraints
        constraints = [tuple(intervals) for _ in range(len(variables))]

        # Initialize problem
        self.problem = Problem(self.expression, self.num_variables, constraints, self.minimize)

        # Initialize a codec to handle genome encodings and decodings
        self.codec = Codec(kind=self.representation, num_variables=self.num_variables,
                           precision=precision, constraint=self.intervals)
        

    def __evaluate(self):

        """
        
        """

        for individual in self.population.get_population():

            # Get phenotype
            phenotype = dict(zip(self.variables, individual.phenotype))

            # Evaluate phenotype
            fitness = self.problem.evaluate(**phenotype)

            # Assign fitness
            individual.fitness = fitness
    
    def __step(self) -> list[Individual]:
        """
        """

        # Unpack rates
        
        # Initialize list for new population
        new_population = []

        while len(new_population) < self.population_size:

            # Select parents
            parents = self.selection(self.population.get_population(), self.minimize)

            # Crossover
            offspring = self.crossover(parents, self.crossover_rate)

            # Mutation
            mutation = self.mutation(offspring, self.mutation_rate, self.intervals)

            # Decode individuals
            for child in offspring:
                phenotype = self.codec.decode(child.genome)
                child.phenotype = phenotype
            
            # Add to new population
            new_population.extend(offspring)

        return new_population
    
    def run(self, population_size: int, num_generations: int,
             show_plot: bool=False, print_console: bool=False) -> tuple[list[float], list[list]]:
        """
        """

        # Store variables
        self.population_size = population_size
        self.num_generations = num_generations

        # Initialize population
        self.population = Population()
        self.population.initialize(self.num_variables, self.intervals, self.codec)

        # Evaluate initial population
        self.__evaluate()

        # Initialize graphs
        graph = Graphics(self.minimize)

        show_plot = (self.num_variables == 2) and show_plot

        if show_plot:
            graph.create_3d_plot(self.population, self.problem, self.intervals)
        
        # Define probabilities
        self.crossover_rate = 0.9
        self.mutation_rate = 1 / (self.codec.num_allels)

        # Run algorithm
        for gen in range(self.num_generations):
            
            # Do an evolutionary step
            new_population = self.__step()

            # Replace population (1 to 1)
            self.population.update_population(new_population)

            # Evaluate population
            self.__evaluate()

            # Plots
            if show_plot:
                graph.update_3d_plot(self.population)
            
            # Save statistics
            best_individual, best_fitness = graph.save_best_individual(self.population)

            if print_console:
                print(f"Generation {gen+1:04d} - Max Fitness: {best_fitness}, Best Phenotype: {best_individual}")

        
        if show_plot:
            graph.plot_fitness()
        
        return graph.fitness_history, graph.best_solutions


class HillClimber():
    
    def __init__(self) -> None:
        pass

    def load_problem(self, expression: str, variables: list[str], intervals: list[float], minimize: bool):

        # Allocate variables
        self.minimize = minimize
        self.expression = expression

        self.sign = 1 if minimize else -1

        self.variables = variables
        self.num_variables = len(self.variables)

        self.intervals = intervals

        # Get constraints
        self.constraints = [tuple(intervals) for _ in range(len(variables))]

        # Initialize problem
        self.problem = Problem(self.expression, self.num_variables, self.constraints, self.minimize)

    def __evaluate(self, x, y):

        z = dict(zip(self.variables, [x,y]))

        return self.problem.evaluate(**z)


    def run(self, num_neighbors: int, max_iterations: int, step_size: float):

        # initial solution
        x = np.random.random(size=(self.num_variables))
        x = x * (self.intervals[1] - self.intervals[0]) + self.intervals[0]

        f = self.__evaluate(x[0], x[1])

        fitness = [f]

        for _ in range(max_iterations):

            # Create neighbors
            neighbors = []
            angle_increment = 2*np.pi / num_neighbors

            for i in range(num_neighbors):

                angle = i * angle_increment

                new_x = float(x[0] + step_size*np.cos(angle))
                new_y = float(x[1] + step_size*np.sin(angle))

                neighbors.append(np.array(new_x, new_y))
            
            # Start a best value as infinity
            best_value = float("inf") if self.minimize else float("-inf")

            # Evaluate neighbors
            for neighbor in neighbors:

                # Evaluate
                new_f = self.__evaluate(neighbor[0], neighbor[1])

                # Verify value
                if self.sign * new_f < best_value:
                    best_value = new_f
                    best_neighbor = neighbor
            
            # Verify if a neighbor is better than the current solution
            if self.sign*best_value < self.sign*f:
                x = best_neighbor
                f = best_value
                fitness.append(f)

            else:
                break

        return fitness, x

class GradientBased(ABC):

    def __init__(self) -> None:
        pass

    def compute_gradient(self, x, y):
        """
        Compute the gradient with numerical derivatives

        Parameters
        ----------
        x: float
            X value
        y: float
            Y value

        Returns
        -------
        np.ndarray
            numpy array with df/dx, df/dy
        """

        dx, dy = 1e-5, 1e-5

        df_dx = (self.__evaluate(x+dx, y) - self.__evaluate(x-dx, y)) / (2*dx)
        df_dy = (self.__evaluate(x, y+dy) - self.__evaluate(x, y-dy)) / (2*dy)

        return np.array([df_dx, df_dy])
    
    def compute_hessian(self, x, y):
        """
        Compute the hessian matrix with numerical derivatives

        Parameters
        ----------
        x: float
            X value
        y: float
            Y value

        Returns
        -------
        np.ndarray
            2D numpy array with hessian
        """

        dx, dy = 1e-5, 1e-5

        df_dxdx = (self.__evaluate(x+dx, y) - 2*self.__evaluate(x, y) + self.__evaluate(x-dx, y)) / (dx**2)
        df_dydy = (self.__evaluate(x, y+dy) - 2*self.__evaluate(x, y) + self.__evaluate(x, y-dy)) / (dy**2)
        df_dxdy = (self.__evaluate(x+dx, y+dx) - self.__evaluate(x+dx, y-dx) - self.__evaluate(x-dx, y+dx) + self.__evaluate(x-dx, y-dx)) / 4*(dx**2)
        df_dydx = (self.__evaluate(x+dy, y+dy) - self.__evaluate(x+dy, y-dy) - self.__evaluate(x-dy, y+dy) + self.__evaluate(x-dy, y-dy)) / 4*(dy**2)
        
        return np.array([[df_dxdx, df_dxdy], [df_dydx, df_dydy]])
    
    def norm(self, point: list[float]) -> float:
        """
        Compute the euclidian norm

        Parameters
        ----------
        point: list[float]
            Point to compute norm
        
        Returns
        -------
        float
        """

        return sum([x**2 for x in point])**0.5
    
    def armijo_condition(self, point, step_size, direction, prev_grad, c1=1e-4):
        """
        """
        new_point = point + step_size*point
        return self.__evaluate(new_point[0], new_point[1]) <= self.__evaluate(point[0], point[1]) + c1 * step_size * np.dot(prev_grad, direction)
    
    def curvature_condition(self, grad_new, direction, prev_grad, c2=0.9):
        """
        """
        return np.dot(grad_new, direction) >= c2 * np.dot(prev_grad, direction)

    def wolfe_conditions(self, point, direction, prev_grad, step_size):
        """
        
        """
        flag = False
        count_iter = 0

        # Until both conditions are fulfilled, or 20 iterations were completed
        while not(flag) and count_iter < 20:

            # Compute new gradient with updated step_size
            grad_new = self.compute_gradient(*(point * direction))

            # If first condition is not fulfilled, decrease step size
            if not(self.armijo_condition(point, step_size, direction, prev_grad)):
                step_size /= 2
            
            # If second condition is not fulfilled, increase step size
            elif not(self.curvature_condition(grad_new, direction, prev_grad)):
                step_size *= 2
            
            else:
                flag = True

        return step_size

    def __evaluate(self, x, y):

        z = dict(zip(self.variables, [x,y]))

        return self.problem.evaluate(**z)


class GradientDescent(GradientBased):
    
    def __init__(self) -> None:
        super().__init__()

    def load_problem(self, expression: str, variables: list[str], intervals: list[float], minimize: bool):

        # Allocate variables
        self.minimize = minimize
        self.expression = expression

        self.variables = variables
        self.num_variables = len(self.variables)

        self.intervals = intervals

        # Get constraints
        self.constraints = [tuple(intervals) for _ in range(len(variables))]

        # Initialize problem
        self.problem = Problem(self.expression, self.num_variables, self.constraints, self.minimize)

    def run(self, step_size: float, tolerance: float):

        # Make an initial solution
        x = np.random.random(size=(self.num_variables))
        x = x * (self.intervals[1] - self.intervals[0]) + self.intervals[0]

        # Make fitness history
        fitness = []

        while self.norm(self.compute_gradient(*x)) > tolerance:

            # Compute a search direction
            grad = self.compute_gradient(*x)
            p = -grad

            step_size = self.wolfe_conditions(x, p, grad, step_size)

            x = x + step_size * p

            fitness.append(self.__evaluate(x[0], x[1]))

        return fitness, x


class NewtonMethod(GradientBased):
    
    def __init__(self) -> None:
        super().__init__()

    def load_problem(self, expression: str, variables: list[str], intervals: list[float], minimize: bool):

        # Allocate variables
        self.minimize = minimize
        self.expression = expression

        self.variables = variables
        self.num_variables = len(self.variables)

        self.intervals = intervals

        # Get constraints
        self.constraints = [tuple(intervals) for _ in range(len(variables))]

        # Initialize problem
        self.problem = Problem(self.expression, self.num_variables, self.constraints, self.minimize)

    def run(self, step_size: float, tolerance: float):

        # Make an initial solution
        x = np.random.random(size=(self.num_variables))
        x = x * (self.intervals[1] - self.intervals[0]) + self.intervals[0]

        # Make fitness history
        fitness = []

        while self.norm(self.compute_gradient(*x)) > tolerance:
            
            # Compute a search direction
            firstDerivative = self.compute_gradient(*x)
            secondDerivative = self.compute_hessian(*x)

            p = -np.matmul(np.linalg.inv(secondDerivative), firstDerivative)

            step_size = self.wolfe_conditions(x, p, firstDerivative, step_size)

            x = x + step_size * p

            fitness.append(self.__evaluate(x[0], x[1]))

        return fitness, x
    