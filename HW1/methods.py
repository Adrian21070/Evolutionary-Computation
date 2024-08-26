
import numpy as np

def hill_climbing(initial_solution, step_size, max_iterations,
                  num_neighbors, function, constraint):
    
    # Evaluate initial solution
    current_solution = initial_solution
    f = function(current_solution, constraint)

    for _ in range(max_iterations):

        # Create neighbors
        neighbors = []
        angle_increment = 2 * np.pi / num_neighbors

        for i in range(num_neighbors):
            # Determine the direction of movement
            angle = i * angle_increment
            # Get new coordinates
            new_x = current_solution[0] + step_size*np.cos(angle)
            new_y = current_solution[1] + step_size*np.sin(angle)
            # Append to the neighbors
            neighbors.append([new_x, new_y])

        # Start a best value as infinity
        best_value = float("inf")

        # Evaluate neighbors
        for neighbor in neighbors:
            # Evaluate
            f = function(neighbor, constraint)

            # Verify value
            if f < best_value:
                best_value = f
                best_neighbor = neighbor

        # Verify if a neighbor is better than current solution
        if function(best_neighbor, constraint) < function(current_solution, constraint):
            current_solution = best_neighbor

        else:
            break

    return current_solution, function(current_solution, constraint)