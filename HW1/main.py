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
    ### Optimal value = [0, 0]

    ## function 2:
    ### Initial solution = [0.5, 1]
    ### constraints = [[-3,3], [-2,2]]
    ### Optimal value = [0.0898, -0.7126]

    ## function 3:
    ### Initial solution = [-2, 2]
    ### constraints = [[-5.12,5.12], [-5.12,5.12]]
    ### Optimal value = [0, 0]

    functions = [function_1, function_2, function_3]
    initial_solutions = [[-4, 4], [0.5, 1], [-2, 2]]
    constraints = [[[-6,6],[-6,6]], [[-3,3],[-2,2]], [[-5.12,5.12],[-5.12,5.12]]]

    optimalValues = [[0, 0], [0.0898, -0.7126], [0, 0]]
    
    for i, function in enumerate(functions):
        
        # Parameters
        initial_solution = initial_solutions[i]

        constraint = constraints[i]

        print(f"Function {i}")

        # Display graph
        ax = plot_contour(function, constraint)

        # HILL CLIMBING
        best_solution, value, iter = hill_climbing(initial_solution, step_size=0.9,
                                             max_iterations=100000, num_neighbors=16,
                                             function=function, constraint=constraint,
                                             ax=ax)
        
        error = np.sqrt((best_solution[0] - optimalValues[i][0])**2 + (best_solution[1] - optimalValues[i][1])**2)

        print("Hill Climbing")
        print("Solution: ", best_solution, "Function: ", value, "Iterations: ", iter, "Error: ", error, "\n\n")

        # GRADIENT DESCENT
        best_solution, value, iter, countGradients = gradient_descent(initial_solution, step_size=0.5,
                                                f=function, constraint=constraint,
                                                tolerance=0.001, ax=ax)
        
        error = np.sqrt((best_solution[0] - optimalValues[i][0])**2 + (best_solution[1] - optimalValues[i][1])**2)

        print("Gradient Descent")
        print("Solution: ", best_solution, "Function: ", value, "Iterations: ", iter, "Error: ", error, "num_Gradients: ", countGradients, "\n\n")

        # NEWTON
        best_solution, value, iter = newton(initial_solution, step_size=0.01,
                                        f=function, constraint=constraint,
                                        tolerance=0.1, ax=ax)
        
        error = np.sqrt((best_solution[0] - optimalValues[i][0])**2 + (best_solution[1] - optimalValues[i][1])**2)

        print("Newton Method")
        print("Solution: ", best_solution, "Function: ", value, "Iterations: ", iter, "Error: ", error, "\n\n")
        

        plt.pause(5)
        plt.clf()