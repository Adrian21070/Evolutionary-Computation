"""
Structure of the project.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from methods import hill_climbing, gradient_descent, newton
from functions import function_1, function_2, function_3


def plot_contour(f, constraints):
    """
    Function to generate the contours.

    @param f: Function to map.
    @param constraints: limits of the functions.
    """

    x = np.linspace(constraints[0][0], constraints[0][1], 100)
    y = np.linspace(constraints[1][0], constraints[1][1], 100)

    X, Y = np.meshgrid(x,y)

    Z = f([X,Y], constraint, is_numpy=True)

    plt.contour(X,Y,Z, cmap="gray", levels=20)
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
    ### Optimal value = [-0.0898, 0.7126] and [0.0898, -0.7126]

    ## function 3:
    ### Initial solution = [-2, 2]
    ### constraints = [[-5.12,5.12], [-5.12,5.12]]
    ### Optimal value = [0, 0]

    # Group functions with their respective parameters
    functions = [function_1, function_2, function_3]
    initial_solutions = [[-4, 4], [0.5, 1], [-2, 2]]
    constraints = [[[-6,6],[-6,6]], [[-3,3],[-2,2]], [[-5.12,5.12],[-5.12,5.12]]]

    # Define parameters per function per algorithm
    step_sizes = [[0.05, 0.05, 0.05], [0.01, 0.01, 0.01], [1.0, 1.0, 1.0]]
    tolerances = [[0, 0.001, 0.001], [0, 0.001, 0.001], [0, 0.001, 0.001]]
    
    # Hill Climbing exclusive
    max_iterations = [1000, 1000, 1000]
    num_neighbors = [4, 4, 8]

    # Know optimal points
    optimalValues = [[0, 0], [-0.0898, 0.7126], [0, 0]]
    
    # For each function ...
    for indx, f in enumerate(functions):

        # Extract parameters
        optimal_solution = np.array(optimalValues[indx])
        initial_solution = initial_solutions[indx]
        constraint = constraints[indx]

        hill_step, grad_step, newton_step = step_sizes[indx]
        _, grad_tolerance, newton_tolerance = tolerances[indx]

        iterations = max_iterations[indx]
        neighbors = num_neighbors[indx]

        print(f"\nFunction {indx+1}")

        # Display Contour
        ax = plot_contour(f, constraint)

        # Perfrom Hill Climbing
        hill_solution, hill_value, hill_total_iterations, hill_total_evaluations = \
            hill_climbing(initial_solution, hill_step, iterations, 
                                    neighbors, f, constraint, ax)
        
        # Compute the error
        hill_error = np.linalg.norm(optimal_solution - np.array(hill_solution))

        print("\nHill Climbing.")
        print(f"Best Solution: {hill_solution}, Function value: {hill_value}, Error: {hill_error}")
        print(f"Num iterations: {hill_total_iterations}, Num evaluations: {hill_total_evaluations}")


        # Perform Gradient Descent
        gradient_solution, gradient_value, gradient_total_iterations, gradient_total_gradients = \
            gradient_descent(initial_solution, grad_step, f,
                              constraint, grad_tolerance, ax)

        # Compute the error
        gradient_error = np.linalg.norm(optimal_solution - np.array(gradient_solution))

        print("\nGradient Descent.")
        print(f"Best Solution: {gradient_solution}, Function value: {gradient_value}, Error: {gradient_error}")
        print(f"Num iterations: {gradient_total_iterations}, Num gradient computations: {gradient_total_gradients}")


        # NEWTON
        newton_solution, newton_value, newton_total_iterations, newton_total_gradients = \
            newton(initial_solution, newton_step, f,
                    constraint, newton_tolerance, ax)

        # Compute the error
        newton_error = np.linalg.norm(optimal_solution - np.array(newton_solution))

        print("\nNewton Method.")
        print(f"Best Solution: {newton_solution}, Function value: {newton_value}, Error: {newton_error}")
        print(f"Num iterations: {newton_total_iterations}, Num gradient computations: {newton_total_gradients}")

        # Create legend labels
        legend_elements = [
                Line2D([0], [0], marker='o', color='r', label='Hill Climbing',
                        markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='b', label='Gradient Descent',
                       markerfacecolor='blue', markersize=10),
                Line2D([0], [0], marker='o', color='g', label='Newton Method',
                       markerfacecolor='green', markersize=10)
            ]
        
        # Add legend
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Wait for 3 seconds
        plt.pause(5)

        plt.savefig(f"Function {indx+1}.png")

        # Close figure
        plt.close()