
import numpy as np
import matplotlib.pyplot as plt

def compute_gradient(x, y, f, constraint):
    dx, dy = 1e-5, 1e-5

    df_dx = (f([x + dx, y], constraint) - f([x - dx, y], constraint)) / (2*dx)
    df_dy = (f([x, y + dy], constraint) - f([x, y - dy], constraint)) / (2*dy)

    return np.array([df_dx, df_dy])

def compute_hessian(x, y, f, constraint):
    dx, dy = 1e-5, 1e-5

    df_dxdx = (f([x + dx, y], constraint) - 2*f([x, y], constraint) + f([x - dx, y], constraint)) / (dx**2)
    df_dydy = (f([x, y + dy], constraint) - 2*f([x, y], constraint) + f([x, y - dy], constraint)) / (dy**2)
    df_dxdy = (f([x + dx, y + dx], constraint) - f([x + dx, y - dx], constraint) - f([x - dx, y + dx], constraint) + f([x - dx, y - dx], constraint)) / 4*(dx**2)
    df_dydx = (f([x + dy, y + dy], constraint) - f([x + dy, y - dy], constraint) - f([x - dy, y + dy], constraint) + f([x - dy, y - dy], constraint)) / 4*(dy**2)
    
    return np.array([[df_dxdx, df_dxdy], [df_dydx, df_dydy]])

def hill_climbing(initial_solution, step_size, max_iterations,
                  num_neighbors, function, constraint, ax):
    
    # Evaluate initial solution
    current_solution = initial_solution
    f = function(current_solution, constraint)

    count_iter = 0
    count_evaluations = 1

    # Plot current solution
    ax.scatter(current_solution[0], current_solution[1], c="r")

    for _ in range(max_iterations):

        # Create neighbors
        neighbors = []
        angle_increment = 2 * np.pi / num_neighbors

        for i in range(num_neighbors):
            # Determine the direction of movement
            angle = i * angle_increment
            # Get new coordinates
            new_x = float(current_solution[0] + step_size*np.cos(angle))
            new_y = float(current_solution[1] + step_size*np.sin(angle))
            # Append to the neighbors
            neighbors.append([new_x, new_y])

        # Start a best value as infinity
        best_value = float("inf")

        # Evaluate neighbors
        for neighbor in neighbors:
            # Evaluate
            f = function(neighbor, constraint)
            count_evaluations += 1

            # Verify value
            if f < best_value:
                best_value = f
                best_neighbor = neighbor

        count_iter += 1

        # Verify if a neighbor is better than current solution
        if function(best_neighbor, constraint) < function(current_solution, constraint):
            current_solution = best_neighbor
            if count_iter % 2 == 0:
                ax.scatter(current_solution[0], current_solution[1], c="r")

        else:
            break
    
        plt.pause(0.1)

    ax.scatter(current_solution[0], current_solution[1], c="r")
    plt.pause(0.1)

    return current_solution, function(current_solution, constraint), count_iter, count_evaluations

def gradient_descent(initial_solution, step_size, f, constraint, tolerance, ax):
    
    x = np.array(initial_solution)

    ax.scatter(x[0], x[1], c="b")

    c1 = 1e-4
    c2 = 0.9

    cont_iter = 0
    countGradients = 0

    while norm(compute_gradient(*x, f, constraint)) > tolerance:
        # Compute a search direction
        grad = compute_gradient(*x, f, constraint)
        p = -grad

        countGradients += 1

        # Compute step size (wolfe conditions)
        # 0 < c1 < c2 < 1, c1 = 10^-4, c2 = 0.9
        # f(x + alpha*p) <= f(x) + c1*alpha*gradient(f)*p
        # gradient(f(x + alpha*p))*p >= c2*gradient(f)*p
        step_size, countGradients = wolfe_conditions(f, x, p, grad, step_size, constraint, countGradients)

        x = x + step_size * p

        if cont_iter % 2 == 0:
            ax.scatter(x[0], x[1], c="b")
            plt.pause(0.1)
            
        cont_iter += 1

    return x, f(x, constraint), cont_iter, countGradients

def armijo_condition(f, point, alpha, p, grad, constraint, c1=1e-4):
    new_point = point + alpha * p
    return f(new_point, constraint) <= f(point, constraint) + c1 * alpha * np.dot(grad, p)
    #return f([x + alpha * p[0], y + alpha * p[1]], constraint) <= f([x,y]) + c1 * alpha * np.dot(grad, p)

def curvature_condition(grad_new, p, grad, c2=0.9):
    return np.dot(grad_new, p) >= c2 * np.dot(grad, p)

def wolfe_conditions(f, point, p, prev_grad, step_size, constraint, countgrad, c1=1e-4, c2=0.9):
    
    flag = False
    count_iter = 0
    
    # Until both conditions are fulfilled, or 20 iterations were completed
    while not(flag) and count_iter < 20:
        
        # Compute new gradient with updated step_size
        grad_new = compute_gradient(point[0] + step_size*p[0], point[1] + step_size*p[1], f, constraint)
        countgrad += 1
        count_iter += 1

        # If first condition is not fulfilled, decrease step size
        if not(armijo_condition(f, point, step_size, p=p, grad=prev_grad, constraint=constraint)):
            step_size /= 2
        
        # If second condition is not fulfilled, increase step size
        elif not(curvature_condition(grad_new, p, prev_grad)):
            step_size *= 2

        else:
            flag = True
    
    return step_size, countgrad

def norm(point: list) -> float:

    return sum([x**2 for x in point])**0.5

def newton(initial_solution, step_size, f, constraint, tolerance, ax):
    
    x = np.array(initial_solution)

    #ax.scatter(x[0], x[1], c="g")

    count_iter = 0
    countGradients = 0

    while norm(compute_gradient(*x, f, constraint)) > tolerance:
        # Compute a search direction
        firstDerivative = compute_gradient(*x, f, constraint)
        secondDerivative = compute_hessian(*x, f, constraint)
        countGradients += 2

        p = - np.matmul(np.linalg.inv(secondDerivative), firstDerivative)
        
        # Compute step size (wolfe conditions)
        # 0 < c1 < c2 < 1, c1 = 10^-4, c2 = 0.9
        # f(x + alpha*p) <= f(x) + c1*alpha*gradient(f)*p
        # gradient(f(x + alpha*p))*p >= c2*gradient(f)*p
        step_size, countGradients = wolfe_conditions(f, x, p, firstDerivative, step_size, constraint, countGradients)

        x = x + step_size * p

        if count_iter % 2 == 0:
            ax.scatter(x[0], x[1], c="g")
            plt.pause(0.1)

        count_iter += 1

    return x, f(x, constraint), count_iter, countGradients