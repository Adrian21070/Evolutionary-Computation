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
    ### Optimal value = [0.0898, -0.7126]

    ## function 3:
    ### Initial solution = [-2, 2]
    ### constraints = [[-5.12,5.12], [-5.12,5.12]]
    ### Optimal value = [0, 0]

    functions = [function_1, function_2, function_3]
    initial_solutions = [[-4, 4], [0.5, 1], [-2, 2]]
    constraints = [[[-6,6],[-6,6]], [[-3,3],[-2,2]], [[-5.12,5.12],[-5.12,5.12]]]

    optimalValues = [[0, 0], [-0.0898, 0.7126], [0, 0]]
    
    function_id = 2

    functions = [functions[function_id]]
    initial_solutions = [initial_solutions[function_id]]
    constraints = [constraints[function_id]]
    optimalValues = [optimalValues[function_id]]

    step_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
    step_sizes = [1.0]
    
    error_hill = []
    error_gradient = []
    error_newton = []

    for step in step_sizes:

        for i, function in enumerate(functions):

            # Parameters
            initial_solution = initial_solutions[i]

            constraint = constraints[i]

            print(f"Function {i}")

            # Display graph
            ax = plot_contour(function, constraint)

            # HILL CLIMBING
            best_solution, value, iter, num_evals = hill_climbing(initial_solution, step_size=step,
                                                 max_iterations=1000, num_neighbors=8,
                                                 function=function, constraint=constraint,
                                                 ax=ax)

            error = np.sqrt((best_solution[0] - optimalValues[i][0])**2 + (best_solution[1] - optimalValues[i][1])**2)
            error_hill.append(error)
            print("Hill Climbing")
            print("Solution: ", best_solution, "Function: ", value, "Iterations: ", iter, "Evaluations: ", num_evals, "Error: ", error, "\n\n")

            # GRADIENT DESCENT
            best_solution, value, iter = gradient_descent(initial_solution, step_size=step,
                                                    f=function, constraint=constraint,
                                                    tolerance=0.001, ax=ax)

            error = np.sqrt((best_solution[0] - optimalValues[i][0])**2 + (best_solution[1] - optimalValues[i][1])**2)

            error_gradient.append(error)

            print("Gradient Descent")
            print("Solution: ", best_solution, "Function: ", value, "Iterations: ", iter, "Error: ", error, "\n\n")

            # NEWTON
            best_solution, value, iter = newton(initial_solution, step_size=step,
                                            f=function, constraint=constraint,
                                            tolerance=0.001, ax=ax)

            error = np.sqrt((best_solution[0] - optimalValues[i][0])**2 + (best_solution[1] - optimalValues[i][1])**2)
            
            error_newton.append(error)

            print("Newton Method")
            print("Solution: ", best_solution, "Function: ", value, "Iterations: ", iter, "Error: ", error, "\n\n")


            plt.pause(1)
            plt.clf()
    
    plt.close()

    fig4, ax4 = plt.subplots(1,1)
    ax4.plot(step_sizes, error_hill, marker='o')
    ax4.set_xticks(step_sizes)
    ax4.set_title("Hill Climbing")

    ax4.set_ylim(0,3)
    ax4.set_xlabel("Step sizes")
    ax4.set_ylabel("Two-Norm Error")

    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(step_sizes, error_gradient, marker='o')
    ax2.set_xticks(step_sizes)
    ax2.set_title("Gradient Descent")

    ax2.set_ylim(0,3)
    ax2.set_xlabel("Step sizes")
    ax2.set_ylabel("Two-Norm Error")

    fig3, ax3 = plt.subplots(1,1)
    ax3.plot(step_sizes, error_newton, marker='o')
    ax3.set_xticks(step_sizes)
    ax3.set_title("Newton Method")

    ax3.set_ylim(0,3)
    ax3.set_xlabel("Step sizes")
    ax3.set_ylabel("Two-Norm Error")

    print(error_newton)

    plt.show()