"""
Structure of the project.
"""

from methods import hill_climbing#, gradient_descent, newton
from functions import function_1, function_2, function_3


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
    initial_solution = [-4,4]
    constraint = [[-6,6],[-6,6]]
    
    best_solution, value = hill_climbing(initial_solution, step_size=0.1, 
                                        max_iterations=1000, num_neighbors=4,
                                        function=f, constraint=constraint)

    print(best_solution)
    print(value)