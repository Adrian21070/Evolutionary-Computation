
import numpy as np

def constraints(solution: list, constraint: list) -> bool:
    """
    @param solution: [x1, x2]
    @param constraint: [[x1_low, x1_high], [x2_low, x2_high]]

    :return: True if it satisfy constraint, else False.
    """

    complete_check = []
    for i in range(len(solution)):
        check = all([solution[i] >= constraint[i][0], solution[i] <= constraint[i][1]])
        complete_check.append(check)
    
    if all(complete_check):
        return True
    
    return False


def function_1(point, constraint, is_numpy=False) -> float:

    # Unpack items
    x1, x2 = point

    if is_numpy:
        # Avoid constraints
        f = -2*(x1**2) + 3*x1*x2 - 1.5*(x2**2) - 1.3
        return -f

    if constraints(point, constraint):
        # Function one needs to be multiplied by -1
        f = -2*(x1**2) + 3*x1*x2 - 1.5*(x2**2) - 1.3
        
        return -f
    else:
        # Outside constraints, penalize it.
        return float("inf")


def function_2(point, constraint, is_numpy=False) -> float:

    # Unpack items
    x1, x2 = point

    if is_numpy:
        # Avoid constraints
        f = (4 - 2.1*(x1**2) + (x1**4)/3)*(x1**2) \
            + x1*x2 + (-4 + 4*(x2**2))*(x2**2)
        
        return f

    if constraints(point, constraint):
        f = (4 - 2.1*(x1**2) + (x1**4)/3)*(x1**2) \
            + x1*x2 + (-4 + 4*(x2**2))*(x2**2)
        
        return f
    else:
        return float("inf")
    

def function_3(point, constraint, is_numpy=False) -> float:

    # Unpack items
    x1, x2 = point

    A = 10

    if is_numpy:
        # Avoid constraints
        f = A + sum([x**2 - A*np.cos(2*np.pi*x) for x in [x1,x2]])

        return f

    if constraints(point, constraint):
        f = A + sum([x**2 - A*np.cos(2*np.pi*x) for x in [x1,x2]])
        
        return f
    else:
        return float("inf")