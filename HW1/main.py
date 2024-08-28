"""
Structure of the project.
"""
import numpy as np
import matplotlib.pyplot as plt

from methods import hill_climbing, gradient_descent, newton
from functions import function_1, function_2, function_3
import os

def plot_contour(f, constraints):
    x = np.linspace(constraints[0][0], constraints[0][1], 100)
    y = np.linspace(constraints[1][0], constraints[1][1], 100)

    X, Y = np.meshgrid(x,y)

    Z = f([X,Y], constraint)

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

    f = function_1

    # Get first derivative
    #x, y = symbols('x y')
    #f_expr = -(-2*(x**2) + 3*x*y - 1.5*(y**2) - 1.3)
    #f_prime = lambdify([x, y], [f_expr.diff(x), f_expr.diff(y)], 'numpy')

    # Parameters
    initial_solution = [0.5,1]
    constraint = [[-3,3],[-2,2]]

    # Display graphic
    #ax = plot_contour(f, constraint)
    
    best_solution, value = hill_climbing(initial_solution, step_size=0.1, 
                                        max_iterations=1000, num_neighbors=4,
                                        function=f, constraint=constraint)
    print(best_solution)
    print(value)

    best_solution, value = gradient_descent(initial_solution, step_size=0.001,
                                            f=f, constraint=constraint, tolerance=0.1)
    print(best_solution)
    print(value)

    best_solution, value = newton(initial_solution, step_size=0.001,
                                            f=f, constraint=constraint, tolerance=0.1)
    print(best_solution)
    print(value)

    plt.show()