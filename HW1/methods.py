
import numpy as np

def compute_gradient(x, y, f, constraint):
    dx, dy = 1e-4, 1e-4

    df_dx = (f([x + dx, y], constraint) - f([x - dx, y], constraint)) / (2*dx)
    df_dy = (f([x, y + dy], constraint) - f([x, y - dy], constraint)) / (2*dy)

    return np.array([df_dx, df_dy])

def compute_secondGradient(x, y, f, constraint):
    dx, dy = 1e-4, 1e-4

    df_dxdx = (f([x + dx, y], constraint) - 2*f([x, y], constraint) + f([x - dx, y], constraint)) / (dx**2)
    df_dydy = (f([x, y + dy], constraint) - 2*f([x, y], constraint) + f([x, y - dy], constraint)) / (dy**2)
    df_dxdy = (f([x + dx, y + dx], constraint) - f([x + dx, y - dx], constraint) - f([x - dx, y + dx], constraint) + f([x - dx, y - dx], constraint)) / 4*(dx**2)
    df_dydx = (f([x + dy, y + dy], constraint) - f([x + dy, y - dy], constraint) - f([x - dy, y + dy], constraint) + f([x - dy, y - dy], constraint)) / 4*(dy**2)
    
    return np.array([[df_dxdx, df_dxdy], [df_dydx, df_dydy]])

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


def gradient_descent(initial_solution, step_size, f, constraint, tolerance):
    
    x = np.array(initial_solution)

    c1 = 1e-4
    c2 = 0.9

    while norm(compute_gradient(*x, f, constraint)) > tolerance:
        # Compute a search direction
        grad = compute_gradient(*x, f, constraint)
        p = -grad

        # Compute step size (wolfe conditions)
        # 0 < c1 < c2 < 1, c1 = 10^-4, c2 = 0.9
        # f(x + alpha*p) <= f(x) + c1*alpha*gradient(f)*p
        # gradient(f(x + alpha*p))*p >= c2*gradient(f)*p
        step_size = wolfe_conditions(f, x, p, grad, step_size, constraint)

        x = x + step_size * p
    
    return x, f(x, constraint)

def armijo_condition(f, point, alpha, p, grad, constraint, c1=1e-4):
    new_point = point + alpha * p
    return f(new_point, constraint) <= f(point, constraint) + c1 * alpha * np.dot(grad, p)
    #return f([x + alpha * p[0], y + alpha * p[1]], constraint) <= f([x,y]) + c1 * alpha * np.dot(grad, p)

def curvature_condition(grad_new, p, grad, c2=0.9):
    return np.dot(grad_new, p) >= c2 * np.dot(grad, p)

def wolfe_conditions(f, point, p, prev_grad, step_size, constraint, c1=1e-4, c2=0.9):
    
    flag = False

    while not(flag):
        
        grad_new = compute_gradient(point[0] + step_size*p[0], point[1] + step_size*p[1], f, constraint)

        if not(armijo_condition(f, point, step_size, p=p, grad=prev_grad, constraint=constraint)):
            step_size /= 2
        elif not(curvature_condition(grad_new, p, prev_grad)):
            step_size *= 2
        else:
            flag = True
    
    return step_size

def norm(point: list) -> float:

    return sum([x**2 for x in point])**0.5

def newton(initial_solution, step_size, f, constraint, tolerance):
    
    x = np.array(initial_solution)

    while norm(compute_gradient(*x, f, constraint)) > tolerance:
        # Compute a search direction
        firstDerivative = compute_gradient(*x, f, constraint)
        secondDerivative = compute_secondGradient(*x, f, constraint)

        p = - np.matmul(np.linalg.inv(secondDerivative), firstDerivative)
        
        # Compute step size (wolfe conditions)
        # 0 < c1 < c2 < 1, c1 = 10^-4, c2 = 0.9
        # f(x + alpha*p) <= f(x) + c1*alpha*gradient(f)*p
        # gradient(f(x + alpha*p))*p >= c2*gradient(f)*p
        step_size = wolfe_conditions(f, x, p, firstDerivative, step_size, constraint)

        x = x + step_size * p
    
    return x, f(x, constraint)