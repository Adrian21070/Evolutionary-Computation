"""
Structure of the project.
"""
import numpy as np
import matplotlib.pyplot as plt

from methods import hill_climbing, gradient_descent, newton
from functions import function_1, function_2, function_3


def plot_contour(f, constraints):
    x = np.linspace(constraints[0][0], constraints[0][1], 100)
    y = np.linspace(constraints[1][0], constraints[1][1], 100)

    X, Y = np.meshgrid(x,y)

    Z = f([X,Y], constraint, is_numpy=True)

    plt.contour(X,Y,Z, levels=20)
    plt.pause(0.1)

    return plt.gca()

if __name__ == "__main__":

    # Notes:
    ## function 1:
    ### Initial solution =[-4, 4]
    ### constraints = [[-6,6], [-6,6]]

    ## function 2:
    ### Initial solution = [0.5, 1]
    ### constraints = [[-3,3], [-2,2]]

    ## function 3:
    ### Initial solution = [-2, 2]
    ### constraints = [[-5.12,5.12], [-5.12,5.12]]

    functions = [function_1, function_2, function_3]
    initial_solutions = [[-4, 4], [0.5, 1], [-2, 2]]
    constraints = [[[-6,6],[-6,6]], [[-3,3],[-2,2]], [[-5.12,5.12],[-5.12,5.12]]]

    for i, function in enumerate(functions):
        
        # Parameters
        initial_solution = initial_solutions[i]

        constraint = constraints[i]

        print(f"Function {i}")

        # Display graph
        ax = plot_contour(function, constraint)

        # HILL CLIMBING
        best_solution, value = hill_climbing(initial_solution, step_size=0.05,
                                             max_iterations=1000, num_neighbors=8,
                                             function=function, constraint=constraint,
                                             ax=ax)
        
        print("Hill Climbing")
        print("Solution: ", best_solution, "Function: ", value, "\n")

        # GRADIENT DESCENT
        best_solution, value = gradient_descent(initial_solution, step_size=0.001,
                                                f=function, constraint=constraint,
                                                tolerance=0.001, ax=ax)

        print("Gradient Descent")
        print("Solution: ", best_solution, "Function: ", value, "\n")

        # NEWTON
        best_solution, value = newton(initial_solution, step_size=0.001,
                                        f=function, constraint=constraint,
                                        tolerance=0.1, ax=ax)
        
        print("Newton Method")
        print("Solution: ", best_solution, "Function: ", value, "\n")

        plt.pause(5)
        plt.clf()