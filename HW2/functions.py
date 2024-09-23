
import numpy as np
import matplotlib.pyplot as plt

# Functions to evaluate
def rastring(*args) -> float | np.ndarray:
    
    A = 10
    n = len(args)

    # If the input is a single scalar, handle it directly
    if all(isinstance(arg, float) for arg in args):
        value = A*n + sum([x**2 - A*np.cos(2*np.pi*x) for x in args])
        return value.item()
    
    # Case with numpy arrays...
    args = np.array(args)
    return A * n + np.sum(args**2 - A * np.cos(2 * np.pi * args), axis=0)
    
    
class Problem():

    def __init__(self, expression: str, num_variables: int,
                constraints: tuple[tuple[float,float], ...], minimize: bool) -> None:
        
        self.expression = expression
        self.num_variables = num_variables
        
        self.minimize = minimize
        self.constraints = constraints

    def evaluate(self, **variables) -> float | np.ndarray:
        
        if not self._check_constraints(variables):
            if self.minimize:
                return float("inf")
            else:
                return float("-inf")
            #raise ValueError("One or more variables do not satisfy the constraints.")

        try:
            # Make the rastring function available for python eval
            local_variables = {"rastring": rastring}
            local_variables.update(variables)

            result = eval(self.expression, {}, local_variables)
            return result
        
        except Exception as e:
            raise ValueError(f"Variable not provided. {e}")
        
    def _check_constraints(self, variables: dict[str, float|np.ndarray]) -> bool:

        for i, (var_name, value) in enumerate(variables.items()):
            if i < len(self.constraints):
                lower, upper = self.constraints[i]

                if isinstance(value, float):
                    # Check scalar values
                    if not(lower <= value <= upper):
                        return False
                elif isinstance(value, np.ndarray):
                    # Check numpy arrays
                    if not(np.all(lower <= value) and np.all(value <= upper)):
                        return False
        return True
