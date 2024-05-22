""""
# 1
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Define the parameters
Ka_pida = 1.0  # PIDA acceleration gain
Kp_pida = 1.0  # PIDA proportional gain
Ki_pida = 1.0  # PIDA integral gain
Kd_pida = 1.0  # PIDA derivative gain
alpha = 1.0    # PIDA filter parameter
beta = 1.0     # PIDA filter parameter

Ka = 10        # Amplifier gain
Ta = 0.1       # Amplifier time constant
Ke = 1         # Exciter gain
Te = 0.4       # Exciter time constant
Kg = 1         # Generator gain
Tg = 1         # Generator time constant
Ks = 1         # Sensor gain
Ts = 0.01      # Sensor time constant

# Define the transfer functions
pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
pida_denominator = [1, alpha, beta, 0]
PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
Exciter = ctrl.TransferFunction([Ke], [Te, 1])
Generator = ctrl.TransferFunction([Kg], [Tg, 1])
Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

# Combine the transfer functions in series for the open-loop system
OpenLoop = PIDA * Amplifier * Exciter * Generator

# Create the closed-loop system with feedback
ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

# Define the time range for the simulation
time = np.linspace(0, 10, 1000)

# Step response of the system
time, response = ctrl.step_response(ClosedLoop, T=time)

# Plot the response
plt.plot(time, response)
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.title('Closed-Loop Step Response with PIDA Controller')
plt.grid(True)
plt.show()
"""
""""
# 2. Grey Wolf Optimizer (GWO)
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl


# Grey Wolf Optimizer (GWO) implementation
def gwo(obj_function, bounds, num_agents=5, max_iter=50):
    dim = len(bounds)
    alpha_pos = np.zeros(dim)
    alpha_score = float('inf')
    beta_pos = np.zeros(dim)
    beta_score = float('inf')
    delta_pos = np.zeros(dim)
    delta_score = float('inf')

    # Initialize the positions of search agents
    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))

    # Main loop
    for iter in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])

            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()

        a = 2 - iter * (2 / max_iter)  # a decreases linearly from 2 to 0

        for i in range(num_agents):
            for j in range(dim):
                r1 = np.random.random()
                r2 = np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3

    return alpha_pos, alpha_score


# Objective function to minimize the integrated absolute error of the step response
def objective_function(params):
    Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = params

    pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
    pida_denominator = [1, alpha, beta, 0]
    PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

    Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
    Exciter = ctrl.TransferFunction([Ke], [Te, 1])
    Generator = ctrl.TransferFunction([Kg], [Tg, 1])
    Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

    OpenLoop = PIDA * Amplifier * Exciter * Generator
    ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

    time = np.linspace(0, 10, 1000)
    time, response = ctrl.step_response(ClosedLoop, T=time)
    error = np.abs(1 - response)

    return np.sum(error)


# Define the parameters
Ka = 10  # Amplifier gain
Ta = 0.1  # Amplifier time constant
Ke = 1  # Exciter gain
Te = 0.4  # Exciter time constant
Kg = 1  # Generator gain
Tg = 1  # Generator time constant
Ks = 1  # Sensor gain
Ts = 0.01  # Sensor time constant

# Define the bounds for the parameters: [Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta]
bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]

# Run the GWO optimization
best_params, best_score = gwo(objective_function, bounds)

# Print the optimized parameters
print("Optimized Parameters:", best_params)
print("Best Score:", best_score)

# Plot the step response with the optimized parameters
Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = best_params

pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
pida_denominator = [1, alpha, beta, 0]
PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
Exciter = ctrl.TransferFunction([Ke], [Te, 1])
Generator = ctrl.TransferFunction([Kg], [Tg, 1])
Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

OpenLoop = PIDA * Amplifier * Exciter * Generator
ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

time = np.linspace(0, 10, 1000)
time, response = ctrl.step_response(ClosedLoop, T=time)

plt.plot(time, response)
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.title('Closed-Loop Step Response with GWO Optimized PIDA Controller')
plt.grid(True)
plt.show()
"""
"""
# 3. Genetic Algorithm (GA)
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from deap import base, creator, tools, algorithms

# Define the system parameters
Ka = 10  # Amplifier gain
Ta = 0.1  # Amplifier time constant
Ke = 1  # Exciter gain
Te = 0.4  # Exciter time constant
Kg = 1  # Generator gain
Tg = 1  # Generator time constant
Ks = 1  # Sensor gain
Ts = 0.01  # Sensor time constant


# Objective function to minimize the integrated absolute error of the step response
def objective_function(params):
    Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = params

    pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
    pida_denominator = [1, alpha, beta, 0]
    PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

    Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
    Exciter = ctrl.TransferFunction([Ke], [Te, 1])
    Generator = ctrl.TransferFunction([Kg], [Tg, 1])
    Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

    OpenLoop = PIDA * Amplifier * Exciter * Generator
    ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

    time = np.linspace(0, 10, 1000)
    _, response = ctrl.step_response(ClosedLoop, T=time)
    error = np.abs(1 - response)

    return np.sum(error),


# Genetic Algorithm implementation using DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", objective_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    population = toolbox.population(n=50)
    ngen = 50
    cxpb = 0.5
    mutpb = 0.2

    result, logbook = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

    best_individual = tools.selBest(result, 1)[0]
    return best_individual


if __name__ == "__main__":
    best_params = main()

    # Print the optimized parameters
    print("Optimized Parameters:", best_params)
    print("Best Score:", objective_function(best_params)[0])

    # Plot the step response with the optimized parameters
    Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = best_params

    pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
    pida_denominator = [1, alpha, beta, 0]
    PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

    Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
    Exciter = ctrl.TransferFunction([Ke], [Te, 1])
    Generator = ctrl.TransferFunction([Kg], [Tg, 1])
    Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

    OpenLoop = PIDA * Amplifier * Exciter * Generator
    ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

    time = np.linspace(0, 10, 1000)
    time, response = ctrl.step_response(ClosedLoop, T=time)

    plt.plot(time, response)
    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.title('Closed-Loop Step Response with GA Optimized PIDA Controller')
    plt.grid(True)
    plt.show()
"""
"""
# 4. Particle Swarm Optimization (PSO)

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from pyswarm import pso


# Objective function to minimize the integrated absolute error of the step response
def objective_function(params):
    Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = params

    pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
    pida_denominator = [1, alpha, beta, 0]
    PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

    Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
    Exciter = ctrl.TransferFunction([Ke], [Te, 1])
    Generator = ctrl.TransferFunction([Kg], [Tg, 1])
    Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

    OpenLoop = PIDA * Amplifier * Exciter * Generator
    ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

    time = np.linspace(0, 10, 1000)
    time, response = ctrl.step_response(ClosedLoop, T=time)
    error = np.abs(1 - response)

    return np.sum(error)


# Define the parameters for the system
Ka = 10  # Amplifier gain
Ta = 0.1  # Amplifier time constant
Ke = 1  # Exciter gain
Te = 0.4  # Exciter time constant
Kg = 1  # Generator gain
Tg = 1  # Generator time constant
Ks = 1  # Sensor gain
Ts = 0.01  # Sensor time constant

# Define the bounds for the parameters: [Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta]
lb = [0, 0, 0, 0, 0, 0]  # Lower bounds
ub = [2, 2, 2, 2, 2, 2]  # Upper bounds

# Run the PSO optimization
best_params, best_score = pso(objective_function, lb, ub, swarmsize=50, maxiter=50)

# Print the optimized parameters
print("Optimized Parameters:", best_params)
print("Best Score:", best_score)

# Plot the step response with the optimized parameters
Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = best_params

pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
pida_denominator = [1, alpha, beta, 0]
PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
Exciter = ctrl.TransferFunction([Ke], [Te, 1])
Generator = ctrl.TransferFunction([Kg], [Tg, 1])
Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

OpenLoop = PIDA * Amplifier * Exciter * Generator
ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

time = np.linspace(0, 10, 1000)
time, response = ctrl.step_response(ClosedLoop, T=time)

plt.plot(time, response)
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.title('Closed-Loop Step Response with PSO Optimized PIDA Controller')
plt.grid(True)
plt.show()
"""
"""
# 5. Benchmarking the Optimization Algorithms
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from pyswarm import pso


# Benchmark Functions
def sphere_function(x):
    return np.sum(x ** 2)


def ackley_function(x):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x ** 2))) - np.exp(0.5 * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e


def rastrigin_function(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


# Objective function to minimize the integrated absolute error of the step response
def objective_function(params, benchmark_func):
    Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = params

    pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
    pida_denominator = [1, alpha, beta, 0]
    PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

    Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
    Exciter = ctrl.TransferFunction([Ke], [Te, 1])
    Generator = ctrl.TransferFunction([Kg], [Tg, 1])
    Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

    OpenLoop = PIDA * Amplifier * Exciter * Generator
    ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

    time = np.linspace(0, 10, 1000)
    time, response = ctrl.step_response(ClosedLoop, T=time)
    error = np.abs(1 - response)

    control_error = np.sum(error)
    benchmark_error = benchmark_func(params)

    return control_error + benchmark_error


# Define the parameters for the system
Ka = 10  # Amplifier gain
Ta = 0.1  # Amplifier time constant
Ke = 1  # Exciter gain
Te = 0.4  # Exciter time constant
Kg = 1  # Generator gain
Tg = 1  # Generator time constant
Ks = 1  # Sensor gain
Ts = 0.01  # Sensor time constant

# Define the bounds for the parameters: [Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta]
lb = [0, 0, 0, 0, 0, 0]  # Lower bounds
ub = [2, 2, 2, 2, 2, 2]  # Upper bounds

# List of benchmark functions
benchmark_functions = [sphere_function, ackley_function, rastrigin_function]
benchmark_names = ["Sphere", "Ackley", "Rastrigin"]

# Optimize PIDA parameters using PSO for each benchmark function
for benchmark_func, benchmark_name in zip(benchmark_functions, benchmark_names):
    print(f"Optimizing with {benchmark_name} Function")
    best_params, best_score = pso(lambda params: objective_function(params, benchmark_func), lb, ub, swarmsize=50,
                                  maxiter=50)

    # Print the optimized parameters
    print(f"Optimized Parameters for {benchmark_name}: {best_params}")
    print(f"Best Score for {benchmark_name}: {best_score}")

    # Plot the step response with the optimized parameters
    Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = best_params

    pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
    pida_denominator = [1, alpha, beta, 0]
    PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

    Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
    Exciter = ctrl.TransferFunction([Ke], [Te, 1])
    Generator = ctrl.TransferFunction([Kg], [Tg, 1])
    Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

    OpenLoop = PIDA * Amplifier * Exciter * Generator
    ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

    time = np.linspace(0, 10, 1000)
    time, response = ctrl.step_response(ClosedLoop, T=time)

    plt.plot(time, response, label=benchmark_name)

plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.title('Closed-Loop Step Response with Optimized PIDA Controller')
plt.legend()
plt.grid(True)
plt.show()
"""
"""
# 6. Benchmarking the Optimization Algorithms

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from deap import base, creator, tools, algorithms
from pyswarm import pso


# Benchmark Functions
def sphere_function(x):
    x = np.array(x)  # Convert to NumPy array
    return np.sum(x ** 2)


def ackley_function(x):
    x = np.array(x)  # Convert to NumPy array
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x ** 2))) - np.exp(0.5 * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e


def rastrigin_function(x):
    x = np.array(x)  # Convert to NumPy array
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


# Objective function to minimize the integrated absolute error of the step response
def objective_function(params, benchmark_func):
    Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = params

    pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
    pida_denominator = [1, alpha, beta, 0]
    PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

    Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
    Exciter = ctrl.TransferFunction([Ke], [Te, 1])
    Generator = ctrl.TransferFunction([Kg], [Tg, 1])
    Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

    OpenLoop = PIDA * Amplifier * Exciter * Generator
    ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

    time = np.linspace(0, 10, 1000)
    time, response = ctrl.step_response(ClosedLoop, T=time)
    error = np.abs(1 - response)

    control_error = np.sum(error)
    benchmark_error = benchmark_func(params)

    return (control_error + benchmark_error,)


# Wrapper for PSO to handle the return type
def pso_objective_wrapper(params, benchmark_func):
    return objective_function(params, benchmark_func)[0]


# Define the parameters for the system
Ka = 10  # Amplifier gain
Ta = 0.1  # Amplifier time constant
Ke = 1  # Exciter gain
Te = 0.4  # Exciter time constant
Kg = 1  # Generator gain
Tg = 1  # Generator time constant
Ks = 1  # Sensor gain
Ts = 0.01  # Sensor time constant

# Define the bounds for the parameters: [Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta]
lb = [0, 0, 0, 0, 0, 0]  # Lower bounds
ub = [2, 2, 2, 2, 2, 2]  # Upper bounds
bounds = list(zip(lb, ub))

# List of benchmark functions
benchmark_functions = [sphere_function, ackley_function, rastrigin_function]
benchmark_names = ["Sphere", "Ackley", "Rastrigin"]


# Define optimization algorithms
def gwo(obj_function, bounds, num_agents=5, max_iter=50):
    dim = len(bounds)
    alpha_pos = np.zeros(dim)
    alpha_score = float('inf')
    beta_pos = np.zeros(dim)
    beta_score = float('inf')
    delta_pos = np.zeros(dim)
    delta_score = float('inf')

    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))

    for iter in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()

        a = 2 - iter * (2 / max_iter)  # a decreases linearly from 2 to 0

        for i in range(num_agents):
            for j in range(dim):
                r1 = np.random.random()
                r2 = np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3

    return alpha_pos, alpha_score


def ga(obj_function, bounds, population_size=50, generations=50):
    dim = len(bounds)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, bounds[0][0], bounds[0][1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", obj_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    result, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)
    best_individual = tools.selBest(result, 1)[0]
    return best_individual, obj_function(best_individual)


# Optimize PIDA parameters using each optimization algorithm for each benchmark function
optimization_algorithms = [pso, ga, gwo]
optimization_names = ["PSO", "GA", "GWO"]

for benchmark_func, benchmark_name in zip(benchmark_functions, benchmark_names):
    for optimization_algorithm, optimization_name in zip(optimization_algorithms, optimization_names):
        print(f"Optimizing with {optimization_name} using {benchmark_name} Function")

        if optimization_algorithm == pso:
            best_params, best_score = pso(lambda params: pso_objective_wrapper(params, benchmark_func), lb, ub,
                                          swarmsize=50, maxiter=50)
        elif optimization_algorithm == ga:
            best_params, best_score = ga(lambda params: objective_function(params, benchmark_func), bounds)
        elif optimization_algorithm == gwo:
            best_params, best_score = gwo(lambda params: objective_function(params, benchmark_func), bounds)

        # Print the optimized parameters
        print(f"Optimized Parameters for {benchmark_name} with {optimization_name}: {best_params}")
        print(f"Best Score for {benchmark_name} with {optimization_name}: {best_score}")

        # Plot the step response with the optimized parameters
        Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = best_params

        pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
        pida_denominator = [1, alpha, beta, 0]
        PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

        Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
        Exciter = ctrl.TransferFunction([Ke], [Te, 1])
        Generator = ctrl.TransferFunction([Kg], [Tg, 1])
        Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

        OpenLoop = PIDA * Amplifier * Exciter * Generator
        ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

        time = np.linspace(0, 10, 1000)
        time, response = ctrl.step_response(ClosedLoop, T=time)

        plt.plot(time, response, label=f"{benchmark_name} with {optimization_name}")

plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.title('Closed-Loop Step Response with Optimized PIDA Controller')
plt.legend()
plt.grid(True)
plt.show()
"""
"""
# 7. Benchmarking the Optimization Algorithms
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from deap import base, creator, tools, algorithms
from pyswarm import pso


# Benchmark Functions
def sphere_function(x):
    x = np.array(x)  # Convert to NumPy array
    return np.sum(x ** 2)


def ackley_function(x):
    x = np.array(x)  # Convert to NumPy array
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x ** 2))) - np.exp(0.5 * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e


def rastrigin_function(x):
    x = np.array(x)  # Convert to NumPy array
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


# Objective function to minimize the integrated absolute error of the step response
def objective_function(params, benchmark_func):
    Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = params

    pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
    pida_denominator = [1, alpha, beta, 0]
    PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

    Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
    Exciter = ctrl.TransferFunction([Ke], [Te, 1])
    Generator = ctrl.TransferFunction([Kg], [Tg, 1])
    Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

    OpenLoop = PIDA * Amplifier * Exciter * Generator
    ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

    time = np.linspace(0, 10, 1000)
    time, response = ctrl.step_response(ClosedLoop, T=time)
    error = np.abs(1 - response)

    control_error = np.sum(error)
    benchmark_error = benchmark_func(params)

    return (control_error + benchmark_error,)


# Wrapper for PSO to handle the return type
def pso_objective_wrapper(params, benchmark_func):
    return objective_function(params, benchmark_func)[0]


# Define the parameters for the system
Ka = 10  # Amplifier gain
Ta = 0.1  # Amplifier time constant
Ke = 1  # Exciter gain
Te = 0.4  # Exciter time constant
Kg = 1  # Generator gain
Tg = 1  # Generator time constant
Ks = 1  # Sensor gain
Ts = 0.01  # Sensor time constant

# Define the bounds for the parameters: [Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta]
lb = [0, 0, 0, 0, 0, 0]  # Lower bounds
ub = [2, 2, 2, 2, 2, 2]  # Upper bounds
bounds = list(zip(lb, ub))

# List of benchmark functions
benchmark_functions = [sphere_function, ackley_function, rastrigin_function]
benchmark_names = ["Sphere", "Ackley", "Rastrigin"]


# Define optimization algorithms
def gwo(obj_function, bounds, num_agents=5, max_iter=50):
    dim = len(bounds)
    alpha_pos = np.zeros(dim)
    alpha_score = float('inf')
    beta_pos = np.zeros(dim)
    beta_score = float('inf')
    delta_pos = np.zeros(dim)
    delta_score = float('inf')

    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))

    for iter in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()

        a = 2 - iter * (2 / max_iter)  # a decreases linearly from 2 to 0

        for i in range(num_agents):
            for j in range(dim):
                r1 = np.random.random()
                r2 = np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3

    return alpha_pos, alpha_score


def ga(obj_function, bounds, population_size=50, generations=50):
    dim = len(bounds)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, bounds[0][0], bounds[0][1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", obj_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    result, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)
    best_individual = tools.selBest(result, 1)[0]
    return best_individual, obj_function(best_individual)


# Optimize PIDA parameters using each optimization algorithm for each benchmark function
optimization_algorithms = [pso, ga, gwo]
optimization_names = ["PSO", "GA", "GWO"]

# Store results for plotting
results = {benchmark_name: {optimization_name: None for optimization_name in optimization_names} for benchmark_name in
           benchmark_names}

for benchmark_func, benchmark_name in zip(benchmark_functions, benchmark_names):
    for optimization_algorithm, optimization_name in zip(optimization_algorithms, optimization_names):
        print(f"Optimizing with {optimization_name} using {benchmark_name} Function")

        if optimization_algorithm == pso:
            best_params, best_score = pso(lambda params: pso_objective_wrapper(params, benchmark_func), lb, ub,
                                          swarmsize=50, maxiter=50)
        elif optimization_algorithm == ga:
            best_params, best_score = ga(lambda params: objective_function(params, benchmark_func), bounds)
        elif optimization_algorithm == gwo:
            best_params, best_score = gwo(lambda params: objective_function(params, benchmark_func), bounds)

        # Print the optimized parameters
        print(f"Optimized Parameters for {benchmark_name} with {optimization_name}: {best_params}")
        print(f"Best Score for {benchmark_name} with {optimization_name}: {best_score}")

        # Store results for plotting
        results[benchmark_name][optimization_name] = best_params

# Plot results for each benchmark function
for benchmark_name in benchmark_names:
    plt.figure()
    for optimization_name in optimization_names:
        best_params = results[benchmark_name][optimization_name]
        Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = best_params

        pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
        pida_denominator = [1, alpha, beta, 0]
        PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

        Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
        Exciter = ctrl.TransferFunction([Ke], [Te, 1])
        Generator = ctrl.TransferFunction([Kg], [Tg, 1])
        Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

        OpenLoop = PIDA * Amplifier * Exciter * Generator
        ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

        time = np.linspace(0, 10, 1000)
        time, response = ctrl.step_response(ClosedLoop, T=time)

        plt.plot(time, response, label=f"{optimization_name}")

    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.title(f'Closed-Loop Step Response with {benchmark_name} Function')
    plt.legend()
    plt.grid(True)
    plt.show()
"""
"""

# 8. Whale Optimization Algorithm (WOA)

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from deap import base, creator, tools, algorithms
from pyswarm import pso


# Benchmark Functions
def sphere_function(x):
    x = np.array(x)  # Convert to NumPy array
    return np.sum(x ** 2)


def ackley_function(x):
    x = np.array(x)  # Convert to NumPy array
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x ** 2))) - np.exp(0.5 * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e


def rastrigin_function(x):
    x = np.array(x)  # Convert to NumPy array
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


# Objective function to minimize the integrated absolute error of the step response
def objective_function(params, benchmark_func):
    Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = params

    pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
    pida_denominator = [1, alpha, beta, 0]
    PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

    Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
    Exciter = ctrl.TransferFunction([Ke], [Te, 1])
    Generator = ctrl.TransferFunction([Kg], [Tg, 1])
    Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

    OpenLoop = PIDA * Amplifier * Exciter * Generator
    ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

    time = np.linspace(0, 10, 1000)
    time, response = ctrl.step_response(ClosedLoop, T=time)
    error = np.abs(1 - response)

    control_error = np.sum(error)
    benchmark_error = benchmark_func(params)

    return (control_error + benchmark_error,)


# Wrapper for PSO to handle the return type
def pso_objective_wrapper(params, benchmark_func):
    return objective_function(params, benchmark_func)[0]


# Define the parameters for the system
Ka = 10  # Amplifier gain
Ta = 0.1  # Amplifier time constant
Ke = 1  # Exciter gain
Te = 0.4  # Exciter time constant
Kg = 1  # Generator gain
Tg = 1  # Generator time constant
Ks = 1  # Sensor gain
Ts = 0.01  # Sensor time constant

# Define the bounds for the parameters: [Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta]
lb = [0, 0, 0, 0, 0, 0]  # Lower bounds
ub = [2, 2, 2, 2, 2, 2]  # Upper bounds
bounds = list(zip(lb, ub))

# List of benchmark functions
benchmark_functions = [sphere_function, ackley_function, rastrigin_function]
benchmark_names = ["Sphere", "Ackley", "Rastrigin"]


# Define optimization algorithms
def gwo(obj_function, bounds, num_agents=5, max_iter=50):
    dim = len(bounds)
    alpha_pos = np.zeros(dim)
    alpha_score = float('inf')
    beta_pos = np.zeros(dim)
    beta_score = float('inf')
    delta_pos = np.zeros(dim)
    delta_score = float('inf')

    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))

    for iter in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()

        a = 2 - iter * (2 / max_iter)  # a decreases linearly from 2 to 0

        for i in range(num_agents):
            for j in range(dim):
                r1 = np.random.random()
                r2 = np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3

    return alpha_pos, alpha_score


def ga(obj_function, bounds, population_size=50, generations=50):
    dim = len(bounds)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, bounds[0][0], bounds[0][1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", obj_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    result, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)
    best_individual = tools.selBest(result, 1)[0]
    return best_individual, obj_function(best_individual)


def woa(obj_function, bounds, num_agents=5, max_iter=50):
    dim = len(bounds)

    # Initialize positions of whales
    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))
    leader_pos = np.zeros(dim)
    leader_score = float('inf')

    for iter in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < leader_score:
                leader_score = fitness
                leader_pos = positions[i, :].copy()

        a = 2 - iter * (2 / max_iter)
        b = 1
        l = np.random.uniform(-1, 1, size=dim)

        for i in range(num_agents):
            r1 = np.random.random()
            r2 = np.random.random()
            A = 2 * a * r1 - a
            C = 2 * r2

            p = np.random.random()
            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * leader_pos - positions[i, :])
                    positions[i, :] = leader_pos - A * D
                else:
                    rand_pos = positions[np.random.randint(0, num_agents), :]
                    D = abs(C * rand_pos - positions[i, :])
                    positions[i, :] = rand_pos - A * D
            else:
                distance_to_leader = abs(leader_pos - positions[i, :])
                positions[i, :] = distance_to_leader * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_pos

    return leader_pos, leader_score


# Optimize PIDA parameters using each optimization algorithm for each benchmark function
optimization_algorithms = [pso, ga, gwo, woa]
optimization_names = ["PSO", "GA", "GWO", "WOA"]

# Store results for plotting
results = {benchmark_name: {optimization_name: None for optimization_name in optimization_names} for benchmark_name in
           benchmark_names}

for benchmark_func, benchmark_name in zip(benchmark_functions, benchmark_names):
    for optimization_algorithm, optimization_name in zip(optimization_algorithms, optimization_names):
        print(f"Optimizing with {optimization_name} using {benchmark_name} Function")

        if optimization_algorithm == pso:
            best_params, best_score = pso(lambda params: pso_objective_wrapper(params, benchmark_func), lb, ub,
                                          swarmsize=50, maxiter=50)
        elif optimization_algorithm == ga:
            best_params, best_score = ga(lambda params: objective_function(params, benchmark_func), bounds)
        elif optimization_algorithm == gwo:
            best_params, best_score = gwo(lambda params: objective_function(params, benchmark_func), bounds)
        elif optimization_algorithm == woa:
            best_params, best_score = woa(lambda params: objective_function(params, benchmark_func), bounds)

        # Print the optimized parameters
        print(f"Optimized Parameters for {benchmark_name} with {optimization_name}: {best_params}")
        print(f"Best Score for {benchmark_name} with {optimization_name}: {best_score}")

        # Store results for plotting
        results[benchmark_name][optimization_name] = best_params

# Plot results for each benchmark function
for benchmark_name in benchmark_names:
    plt.figure()
    for optimization_name in optimization_names:
        best_params = results[benchmark_name][optimization_name]
        Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = best_params

        pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
        pida_denominator = [1, alpha, beta, 0]
        PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

        Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
        Exciter = ctrl.TransferFunction([Ke], [Te, 1])
        Generator = ctrl.TransferFunction([Kg], [Tg, 1])
        Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

        OpenLoop = PIDA * Amplifier * Exciter * Generator
        ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

        time = np.linspace(0, 10, 1000)
        time, response = ctrl.step_response(ClosedLoop, T=time)

        plt.plot(time, response, label=f"{optimization_name}")

    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.title(f'Closed-Loop Step Response with {benchmark_name} Function')
    plt.legend()
    plt.grid(True)
    plt.show()
"""
"""
# 9. convergence

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from deap import base, creator, tools, algorithms
from pyswarm import pso


# Benchmark Functions
def sphere_function(x):
    x = np.array(x)  # Convert to NumPy array
    return np.sum(x ** 2)


def ackley_function(x):
    x = np.array(x)  # Convert to NumPy array
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x ** 2))) - np.exp(0.5 * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e


def rastrigin_function(x):
    x = np.array(x)  # Convert to NumPy array
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


# Objective function to minimize the integrated absolute error of the step response
def objective_function(params, benchmark_func):
    Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = params

    pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
    pida_denominator = [1, alpha, beta, 0]
    PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

    Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
    Exciter = ctrl.TransferFunction([Ke], [Te, 1])
    Generator = ctrl.TransferFunction([Kg], [Tg, 1])
    Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

    OpenLoop = PIDA * Amplifier * Exciter * Generator
    ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

    time = np.linspace(0, 10, 1000)
    time, response = ctrl.step_response(ClosedLoop, T=time)
    error = np.abs(1 - response)

    control_error = np.sum(error)
    benchmark_error = benchmark_func(params)

    return (control_error + benchmark_error,)


# Define the parameters for the system
Ka = 10  # Amplifier gain
Ta = 0.1  # Amplifier time constant
Ke = 1  # Exciter gain
Te = 0.4  # Exciter time constant
Kg = 1  # Generator gain
Tg = 1  # Generator time constant
Ks = 1  # Sensor gain
Ts = 0.01  # Sensor time constant

# Define the bounds for the parameters: [Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta]
lb = [0, 0, 0, 0, 0, 0]  # Lower bounds
ub = [2, 2, 2, 2, 2, 2]  # Upper bounds
bounds = list(zip(lb, ub))

# List of benchmark functions
benchmark_functions = [sphere_function, ackley_function, rastrigin_function]
benchmark_names = ["Sphere", "Ackley", "Rastrigin"]


def gwo_algorithm(obj_function, bounds, num_agents=5, max_iter=50):
    dim = len(bounds)
    alpha_pos = np.zeros(dim)
    alpha_score = float('inf')
    beta_pos = np.zeros(dim)
    beta_score = float('inf')
    delta_pos = np.zeros(dim)
    delta_score = float('inf')

    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))
    convergence_curve = []

    for iter in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()

        a = 2 - iter * (2 / max_iter)  # a decreases linearly from 2 to 0

        for i in range(num_agents):
            for j in range(dim):
                r1 = np.random.random()
                r2 = np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3

        convergence_curve.append(alpha_score)

    return alpha_pos, alpha_score, convergence_curve


def ga_algorithm(obj_function, bounds, population_size=50, generations=50):
    dim = len(bounds)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, bounds[0][0], bounds[0][1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", obj_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    logbook = tools.Logbook()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    result, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, stats=stats,
                                          verbose=False)
    best_individual = tools.selBest(result, 1)[0]

    convergence_curve = [entry['min'] for entry in logbook]

    return best_individual, obj_function(best_individual), convergence_curve, logbook


def woa_algorithm(obj_function, bounds, num_agents=5, max_iter=50):
    dim = len(bounds)

    # Initialize positions of whales
    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))
    leader_pos = np.zeros(dim)
    leader_score = float('inf')

    convergence_curve = []

    for iter in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < leader_score:
                leader_score = fitness
                leader_pos = positions[i, :].copy()

        a = 2 - iter * (2 / max_iter)
        b = 1
        l = np.random.uniform(-1, 1, size=dim)

        for i in range(num_agents):
            r1 = np.random.random()
            r2 = np.random.random()
            A = 2 * a * r1 - a
            C = 2 * r2

            p = np.random.random()
            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * leader_pos - positions[i, :])
                    positions[i, :] = leader_pos - A * D
                else:
                    rand_pos = positions[np.random.randint(0, num_agents), :]
                    D = abs(C * rand_pos - positions[i, :])
                    positions[i, :] = rand_pos - A * D
            else:
                distance_to_leader = abs(leader_pos - positions[i, :])
                positions[i, :] = distance_to_leader * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_pos

        convergence_curve.append(leader_score)

    return leader_pos, leader_score, convergence_curve


# Optimize PIDA parameters using each optimization algorithm for each benchmark function
optimization_algorithms = [ ga_algorithm, gwo_algorithm, woa_algorithm]
optimization_names = ["GA", "GWO", "WOA"]

# Store results for plotting
results = {benchmark_name: {optimization_name: None for optimization_name in optimization_names} for benchmark_name in
           benchmark_names}
convergence_curves = {benchmark_name: {optimization_name: None for optimization_name in optimization_names} for
                      benchmark_name in benchmark_names}
stats_logs = {benchmark_name: {optimization_name: None for optimization_name in optimization_names} for benchmark_name
              in benchmark_names}

for benchmark_func, benchmark_name in zip(benchmark_functions, benchmark_names):
    for optimization_algorithm, optimization_name in zip(optimization_algorithms, optimization_names):
        print(f"Optimizing with {optimization_name} using {benchmark_name} Function")


        if   optimization_algorithm == ga_algorithm:
            best_params, best_score, convergence_curve, logbook = ga_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
        elif optimization_algorithm == gwo_algorithm:
            best_params, best_score, convergence_curve = gwo_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
            logbook = None  # GWO doesn't provide logbook by default
        elif optimization_algorithm == woa_algorithm:
            best_params, best_score, convergence_curve = woa_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
            logbook = None  # WOA doesn't provide logbook by default

        # Print the optimized parameters
        print(f"Optimized Parameters for {benchmark_name} with {optimization_name}: {best_params}")
        print(f"Best Score for {benchmark_name} with {optimization_name}: {best_score}")

        # Store results for plotting
        results[benchmark_name][optimization_name] = best_params
        convergence_curves[benchmark_name][optimization_name] = convergence_curve
        stats_logs[benchmark_name][optimization_name] = logbook

# Plot results for each benchmark function
for benchmark_name in benchmark_names:
    plt.figure()
    for optimization_name in optimization_names:
        best_params = results[benchmark_name][optimization_name]
        Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = best_params

        pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
        pida_denominator = [1, alpha, beta, 0]
        PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

        Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
        Exciter = ctrl.TransferFunction([Ke], [Te, 1])
        Generator = ctrl.TransferFunction([Kg], [Tg, 1])
        Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

        OpenLoop = PIDA * Amplifier * Exciter * Generator
        ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

        time = np.linspace(0, 10, 1000)
        time, response = ctrl.step_response(ClosedLoop, T=time)

        plt.plot(time, response, label=f"{optimization_name}")

    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.title(f'Closed-Loop Step Response with {benchmark_name} Function')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot convergence curves and stats for each benchmark function
for benchmark_name in benchmark_names:
    plt.figure(figsize=(14, 8))

    # Plot convergence curves

    for optimization_name in optimization_names:
        convergence_curve = convergence_curves[benchmark_name][optimization_name]
        if convergence_curve is not None:
            plt.plot(convergence_curve, label=f"{optimization_name}")
    plt.xlabel('Iterations')
    plt.ylabel('Best Score')
    plt.title(f'Convergence Curve with {benchmark_name} Function')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
"""
"""
# 10. HHO

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from deap import base, creator, tools, algorithms
from pyswarm import pso

# Benchmark Functions
def sphere_function(x):
    x = np.array(x)  # Convert to NumPy array
    return np.sum(x ** 2)

def ackley_function(x):
    x = np.array(x)  # Convert to NumPy array
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x ** 2))) - np.exp(0.5 * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e

def rastrigin_function(x):
    x = np.array(x)  # Convert to NumPy array
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

# Objective function to minimize the integrated absolute error of the step response
def objective_function(params, benchmark_func):
    Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = params

    pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
    pida_denominator = [1, alpha, beta, 0]
    PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

    Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
    Exciter = ctrl.TransferFunction([Ke], [Te, 1])
    Generator = ctrl.TransferFunction([Kg], [Tg, 1])
    Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

    OpenLoop = PIDA * Amplifier * Exciter * Generator
    ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

    time = np.linspace(0, 10, 1000)
    time, response = ctrl.step_response(ClosedLoop, T=time)
    error = np.abs(1 - response)

    control_error = np.sum(error)
    benchmark_error = benchmark_func(params)

    return (control_error + benchmark_error,)

# Define the parameters for the system
Ka = 10  # Amplifier gain
Ta = 0.1  # Amplifier time constant
Ke = 1  # Exciter gain
Te = 0.4  # Exciter time constant
Kg = 1  # Generator gain
Tg = 1  # Generator time constant
Ks = 1  # Sensor gain
Ts = 0.01  # Sensor time constant

# Define the bounds for the parameters: [Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta]
lb = [0, 0, 0, 0, 0, 0]  # Lower bounds
ub = [2, 2, 2, 2, 2, 2]  # Upper bounds
bounds = list(zip(lb, ub))

# List of benchmark functions
benchmark_functions = [sphere_function, ackley_function, rastrigin_function]
benchmark_names = ["Sphere", "Ackley", "Rastrigin"]

def gwo_algorithm(obj_function, bounds, num_agents=5, max_iter=50):
    dim = len(bounds)
    alpha_pos = np.zeros(dim)
    alpha_score = float('inf')
    beta_pos = np.zeros(dim)
    beta_score = float('inf')
    delta_pos = np.zeros(dim)
    delta_score = float('inf')

    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))
    convergence_curve = []

    for iter in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()

        a = 2 - iter * (2 / max_iter)  # a decreases linearly from 2 to 0

        for i in range(num_agents):
            for j in range(dim):
                r1 = np.random.random()
                r2 = np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3

        convergence_curve.append(alpha_score)

    return alpha_pos, alpha_score, convergence_curve

def ga_algorithm(obj_function, bounds, population_size=50, generations=50):
    dim = len(bounds)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, bounds[0][0], bounds[0][1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", obj_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    logbook = tools.Logbook()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    result, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, stats=stats,
                                          verbose=False)
    best_individual = tools.selBest(result, 1)[0]

    convergence_curve = [entry['min'] for entry in logbook]

    return best_individual, obj_function(best_individual), convergence_curve, logbook

def woa_algorithm(obj_function, bounds, num_agents=5, max_iter=50):
    dim = len(bounds)

    # Initialize positions of whales
    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))
    leader_pos = np.zeros(dim)
    leader_score = float('inf')

    convergence_curve = []

    for iter in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < leader_score:
                leader_score = fitness
                leader_pos = positions[i, :].copy()

        a = 2 - iter * (2 / max_iter)
        b = 1
        l = np.random.uniform(-1, 1, size=dim)

        for i in range(num_agents):
            r1 = np.random.random()
            r2 = np.random.random()
            A = 2 * a * r1 - a
            C = 2 * r2

            p = np.random.random()
            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * leader_pos - positions[i, :])
                    positions[i, :] = leader_pos - A * D
                else:
                    rand_pos = positions[np.random.randint(0, num_agents), :]
                    D = abs(C * rand_pos - positions[i, :])
                    positions[i, :] = rand_pos - A * D
            else:
                distance_to_leader = abs(leader_pos - positions[i, :])
                positions[i, :] = distance_to_leader * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_pos

        convergence_curve.append(leader_score)

    return leader_pos, leader_score, convergence_curve

def hho_algorithm(obj_function, bounds, num_agents=5, max_iter=50):
    dim = len(bounds)
    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))
    best_position = np.zeros(dim)
    best_score = float('inf')
    convergence_curve = []

    for t in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < best_score:
                best_score = fitness
                best_position = positions[i, :].copy()

        E1 = 2 * (1 - t / max_iter)
        for i in range(num_agents):
            E0 = 2 * np.random.random() - 1
            E = E1 * E0
            q = np.random.random()
            r = np.random.random()

            if abs(E) >= 1:
                rand_index = np.random.randint(0, num_agents)
                X_rand = positions[rand_index, :]
                positions[i, :] = X_rand - r * abs(X_rand - 2 * r * positions[i, :])
            elif q < 0.5:
                if abs(E) < 0.5:
                    positions[i, :] = best_position - E * abs(best_position - positions[i, :])
                else:
                    positions[i, :] = best_position - E * abs(best_position - positions[i, :]) + np.random.randn(dim) * 0.01
            else:
                D_best = abs(best_position - positions[i, :])
                positions[i, :] = D_best * np.exp(E * (np.random.random() - 1)) * np.cos(E * 2 * np.pi) + best_position

        convergence_curve.append(best_score)

    return best_position, best_score, convergence_curve

# Optimize PIDA parameters using each optimization algorithm for each benchmark function
optimization_algorithms = [ga_algorithm, gwo_algorithm, woa_algorithm, hho_algorithm]
optimization_names = ["GA", "GWO", "WOA", "HHO"]

# Store results for plotting
results = {benchmark_name: {optimization_name: None for optimization_name in optimization_names} for benchmark_name in
           benchmark_names}
convergence_curves = {benchmark_name: {optimization_name: None for optimization_name in optimization_names} for
                      benchmark_name in benchmark_names}
stats_logs = {benchmark_name: {optimization_name: None for optimization_name in optimization_names} for benchmark_name
              in benchmark_names}

for benchmark_func, benchmark_name in zip(benchmark_functions, benchmark_names):
    for optimization_algorithm, optimization_name in zip(optimization_algorithms, optimization_names):
        print(f"Optimizing with {optimization_name} using {benchmark_name} Function")

        if optimization_algorithm == ga_algorithm:
            best_params, best_score, convergence_curve, logbook = ga_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
        elif optimization_algorithm == gwo_algorithm:
            best_params, best_score, convergence_curve = gwo_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
            logbook = None  # GWO doesn't provide logbook by default
        elif optimization_algorithm == woa_algorithm:
            best_params, best_score, convergence_curve = woa_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
            logbook = None  # WOA doesn't provide logbook by default
        elif optimization_algorithm == hho_algorithm:
            best_params, best_score, convergence_curve = hho_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
            logbook = None  # HHO doesn't provide logbook by default

        # Print the optimized parameters
        print(f"Optimized Parameters for {benchmark_name} with {optimization_name}: {best_params}")
        print(f"Best Score for {benchmark_name} with {optimization_name}: {best_score}")

        # Store results for plotting
        results[benchmark_name][optimization_name] = best_params
        convergence_curves[benchmark_name][optimization_name] = convergence_curve
        stats_logs[benchmark_name][optimization_name] = logbook

# Plot results for each benchmark function
for benchmark_name in benchmark_names:
    plt.figure()
    for optimization_name in optimization_names:
        best_params = results[benchmark_name][optimization_name]
        Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = best_params

        pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
        pida_denominator = [1, alpha, beta, 0]
        PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

        Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
        Exciter = ctrl.TransferFunction([Ke], [Te, 1])
        Generator = ctrl.TransferFunction([Kg], [Tg, 1])
        Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

        OpenLoop = PIDA * Amplifier * Exciter * Generator
        ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

        time = np.linspace(0, 10, 1000)
        time, response = ctrl.step_response(ClosedLoop, T=time)

        plt.plot(time, response, label=f"{optimization_name}")

    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.title(f'Closed-Loop Step Response with {benchmark_name} Function')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot convergence curves and stats for each benchmark function
for benchmark_name in benchmark_names:
    plt.figure(figsize=(14, 8))

    # Plot convergence curves
    for optimization_name in optimization_names:
        convergence_curve = convergence_curves[benchmark_name][optimization_name]
        if convergence_curve is not None:
            plt.plot(convergence_curve, label=f"{optimization_name}")
    plt.xlabel('Iterations')
    plt.ylabel('Best Score')
    plt.title(f'Convergence Curve with {benchmark_name} Function')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
"""
"""
# 11. Golden Eagle Optimization

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from deap import base, creator, tools, algorithms
from pyswarm import pso

# Benchmark Functions
def sphere_function(x):
    x = np.array(x)  # Convert to NumPy array
    return np.sum(x ** 2)

def ackley_function(x):
    x = np.array(x)  # Convert to NumPy array
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x ** 2))) - np.exp(0.5 * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e

def rastrigin_function(x):
    x = np.array(x)  # Convert to NumPy array
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

# Objective function to minimize the integrated absolute error of the step response
def objective_function(params, benchmark_func):
    Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = params

    pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
    pida_denominator = [1, alpha, beta, 0]
    PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

    Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
    Exciter = ctrl.TransferFunction([Ke], [Te, 1])
    Generator = ctrl.TransferFunction([Kg], [Tg, 1])
    Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

    OpenLoop = PIDA * Amplifier * Exciter * Generator
    ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

    time = np.linspace(0, 10, 1000)
    time, response = ctrl.step_response(ClosedLoop, T=time)
    error = np.abs(1 - response)

    control_error = np.sum(error)
    benchmark_error = benchmark_func(params)

    return (control_error + benchmark_error,)

# Define the parameters for the system
Ka = 10  # Amplifier gain
Ta = 0.1  # Amplifier time constant
Ke = 1  # Exciter gain
Te = 0.4  # Exciter time constant
Kg = 1  # Generator gain
Tg = 1  # Generator time constant
Ks = 1  # Sensor gain
Ts = 0.01  # Sensor time constant

# Define the bounds for the parameters: [Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta]
lb = [0, 0, 0, 0, 0, 0]  # Lower bounds
ub = [2, 2, 2, 2, 2, 2]  # Upper bounds
bounds = list(zip(lb, ub))

# List of benchmark functions
benchmark_functions = [sphere_function, ackley_function, rastrigin_function]
benchmark_names = ["Sphere", "Ackley", "Rastrigin"]

def gwo_algorithm(obj_function, bounds, num_agents=5, max_iter=50):
    dim = len(bounds)
    alpha_pos = np.zeros(dim)
    alpha_score = float('inf')
    beta_pos = np.zeros(dim)
    beta_score = float('inf')
    delta_pos = np.zeros(dim)
    delta_score = float('inf')

    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))
    convergence_curve = []

    for iter in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()

        a = 2 - iter * (2 / max_iter)  # a decreases linearly from 2 to 0

        for i in range(num_agents):
            for j in range(dim):
                r1 = np.random.random()
                r2 = np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3

        convergence_curve.append(alpha_score)

    return alpha_pos, alpha_score, convergence_curve

def ga_algorithm(obj_function, bounds, population_size=50, generations=50):
    dim = len(bounds)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, bounds[0][0], bounds[0][1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", obj_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    logbook = tools.Logbook()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    result, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, stats=stats,
                                          verbose=False)
    best_individual = tools.selBest(result, 1)[0]

    convergence_curve = [entry['min'] for entry in logbook]

    return best_individual, obj_function(best_individual), convergence_curve, logbook

def woa_algorithm(obj_function, bounds, num_agents=5, max_iter=50):
    dim = len(bounds)

    # Initialize positions of whales
    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))
    leader_pos = np.zeros(dim)
    leader_score = float('inf')

    convergence_curve = []

    for iter in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < leader_score:
                leader_score = fitness
                leader_pos = positions[i, :].copy()

        a = 2 - iter * (2 / max_iter)
        b = 1
        l = np.random.uniform(-1, 1, size=dim)

        for i in range(num_agents):
            r1 = np.random.random()
            r2 = np.random.random()
            A = 2 * a * r1 - a
            C = 2 * r2

            p = np.random.random()
            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * leader_pos - positions[i, :])
                    positions[i, :] = leader_pos - A * D
                else:
                    rand_pos = positions[np.random.randint(0, num_agents), :]
                    D = abs(C * rand_pos - positions[i, :])
                    positions[i, :] = rand_pos - A * D
            else:
                distance_to_leader = abs(leader_pos - positions[i, :])
                positions[i, :] = distance_to_leader * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_pos

        convergence_curve.append(leader_score)

    return leader_pos, leader_score, convergence_curve

def hho_algorithm(obj_function, bounds, num_agents=5, max_iter=50):
    dim = len(bounds)
    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))
    best_position = np.zeros(dim)
    best_score = float('inf')
    convergence_curve = []

    for t in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < best_score:
                best_score = fitness
                best_position = positions[i, :].copy()

        E1 = 2 * (1 - t / max_iter)
        for i in range(num_agents):
            E0 = 2 * np.random.random() - 1
            E = E1 * E0
            q = np.random.random()
            r = np.random.random()

            if abs(E) >= 1:
                rand_index = np.random.randint(0, num_agents)
                X_rand = positions[rand_index, :]
                positions[i, :] = X_rand - r * abs(X_rand - 2 * r * positions[i, :])
            elif q < 0.5:
                if abs(E) < 0.5:
                    positions[i, :] = best_position - E * abs(best_position - positions[i, :])
                else:
                    positions[i, :] = best_position - E * abs(best_position - positions[i, :]) + np.random.randn(dim) * 0.01
            else:
                D_best = abs(best_position - positions[i, :])
                positions[i, :] = D_best * np.exp(E * (np.random.random() - 1)) * np.cos(E * 2 * np.pi) + best_position

        convergence_curve.append(best_score)

    return best_position, best_score, convergence_curve

def geo_algorithm(obj_function, bounds, num_agents=5, max_iter=50):
    dim = len(bounds)
    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))
    best_position = np.zeros(dim)
    best_score = float('inf')
    convergence_curve = []

    for t in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < best_score:
                best_score = fitness
                best_position = positions[i, :].copy()

        for i in range(num_agents):
            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()
            r4 = np.random.random()
            if r1 < 0.5:
                if r2 < 0.5:
                    positions[i, :] = positions[i, :] + r3 * (best_position - abs(positions[i, :]))
                else:
                    positions[i, :] = best_position - r3 * abs(positions[i, :] - best_position)
            else:
                if r2 < 0.5:
                    positions[i, :] = positions[i, :] + r3 * (best_position - abs(positions[i, :])) + r4 * (positions[i, :] - best_position)
                else:
                    positions[i, :] = positions[i, :] - r3 * (best_position - abs(positions[i, :])) - r4 * (positions[i, :] - best_position)

        convergence_curve.append(best_score)

    return best_position, best_score, convergence_curve

# Optimize PIDA parameters using each optimization algorithm for each benchmark function
optimization_algorithms = [ga_algorithm, gwo_algorithm, woa_algorithm, hho_algorithm, geo_algorithm]
optimization_names = ["GA", "GWO", "WOA", "HHO", "GEO"]

# Store results for plotting
results = {benchmark_name: {optimization_name: None for optimization_name in optimization_names} for benchmark_name in
           benchmark_names}
convergence_curves = {benchmark_name: {optimization_name: None for optimization_name in optimization_names} for
                      benchmark_name in benchmark_names}
stats_logs = {benchmark_name: {optimization_name: None for optimization_name in optimization_names} for benchmark_name
              in benchmark_names}

for benchmark_func, benchmark_name in zip(benchmark_functions, benchmark_names):
    for optimization_algorithm, optimization_name in zip(optimization_algorithms, optimization_names):
        print(f"Optimizing with {optimization_name} using {benchmark_name} Function")

        if optimization_algorithm == ga_algorithm:
            best_params, best_score, convergence_curve, logbook = ga_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
        elif optimization_algorithm == gwo_algorithm:
            best_params, best_score, convergence_curve = gwo_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
            logbook = None  # GWO doesn't provide logbook by default
        elif optimization_algorithm == woa_algorithm:
            best_params, best_score, convergence_curve = woa_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
            logbook = None  # WOA doesn't provide logbook by default
        elif optimization_algorithm == hho_algorithm:
            best_params, best_score, convergence_curve = hho_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
            logbook = None  # HHO doesn't provide logbook by default
        elif optimization_algorithm == geo_algorithm:
            best_params, best_score, convergence_curve = geo_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
            logbook = None  # GEO doesn't provide logbook by default

        # Print the optimized parameters
        print(f"Optimized Parameters for {benchmark_name} with {optimization_name}: {best_params}")
        print(f"Best Score for {benchmark_name} with {optimization_name}: {best_score}")

        # Store results for plotting
        results[benchmark_name][optimization_name] = best_params
        convergence_curves[benchmark_name][optimization_name] = convergence_curve
        stats_logs[benchmark_name][optimization_name] = logbook

# Plot results for each benchmark function
for benchmark_name in benchmark_names:
    plt.figure()
    for optimization_name in optimization_names:
        best_params = results[benchmark_name][optimization_name]
        Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = best_params

        pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
        pida_denominator = [1, alpha, beta, 0]
        PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

        Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
        Exciter = ctrl.TransferFunction([Ke], [Te, 1])
        Generator = ctrl.TransferFunction([Kg], [Tg, 1])
        Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

        OpenLoop = PIDA * Amplifier * Exciter * Generator
        ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

        time = np.linspace(0, 10, 1000)
        time, response = ctrl.step_response(ClosedLoop, T=time)

        plt.plot(time, response, label=f"{optimization_name}")

    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.title(f'Closed-Loop Step Response with {benchmark_name} Function')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot convergence curves and stats for each benchmark function
for benchmark_name in benchmark_names:
    plt.figure(figsize=(14, 8))

    # Plot convergence curves
    for optimization_name in optimization_names:
        convergence_curve = convergence_curves[benchmark_name][optimization_name]
        if convergence_curve is not None:
            plt.plot(convergence_curve, label=f"{optimization_name}")
    plt.xlabel('Iterations')
    plt.ylabel('Best Score')
    plt.title(f'Convergence Curve with {benchmark_name} Function')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
"""

# 12. Test Functions

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from deap import base, creator, tools, algorithms
from pyswarm import pso

# Benchmark Functions
def sphere_function(x):
    x = np.array(x)  # Convert to NumPy array
    return np.sum(x ** 2)

def ackley_function(x):
    x = np.array(x)  # Convert to NumPy array
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x ** 2))) - np.exp(0.5 * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e

def rastrigin_function(x):
    x = np.array(x)  # Convert to NumPy array
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock_function(x):
    x = np.array(x)  # Convert to NumPy array
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def schwefel_function(x):
    x = np.array(x)  # Convert to NumPy array
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def griewank_function(x):
    x = np.array(x)  # Convert to NumPy array
    return 1 + np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

# Objective function to minimize the integrated absolute error of the step response
def objective_function(params, benchmark_func):
    Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = params

    pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
    pida_denominator = [1, alpha, beta, 0]
    PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

    Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
    Exciter = ctrl.TransferFunction([Ke], [Te, 1])
    Generator = ctrl.TransferFunction([Kg], [Tg, 1])
    Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

    OpenLoop = PIDA * Amplifier * Exciter * Generator
    ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

    time = np.linspace(0, 10, 1000)
    time, response = ctrl.step_response(ClosedLoop, T=time)
    error = np.abs(1 - response)

    control_error = np.sum(error)
    benchmark_error = benchmark_func(params)

    return (control_error + benchmark_error,)

# Define the parameters for the system
Ka = 10  # Amplifier gain
Ta = 0.1  # Amplifier time constant
Ke = 1  # Exciter gain
Te = 0.4  # Exciter time constant
Kg = 1  # Generator gain
Tg = 1  # Generator time constant
Ks = 1  # Sensor gain
Ts = 0.01  # Sensor time constant

# Define the bounds for the parameters: [Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta]
lb = [0, 0, 0, 0, 0, 0]  # Lower bounds
ub = [2, 2, 2, 2, 2, 2]  # Upper bounds
bounds = list(zip(lb, ub))

# List of benchmark functions
benchmark_functions = [sphere_function, ackley_function, rastrigin_function, rosenbrock_function, schwefel_function, griewank_function]
benchmark_names = ["Sphere", "Ackley", "Rastrigin", "Rosenbrock", "Schwefel", "Griewank"]

def gwo_algorithm(obj_function, bounds, num_agents=5, max_iter=100):
    dim = len(bounds)
    alpha_pos = np.zeros(dim)
    alpha_score = float('inf')
    beta_pos = np.zeros(dim)
    beta_score = float('inf')
    delta_pos = np.zeros(dim)
    delta_score = float('inf')

    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))
    convergence_curve = []

    for iter in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()

        a = 2 - iter * (2 / max_iter)  # a decreases linearly from 2 to 0

        for i in range(num_agents):
            for j in range(dim):
                r1 = np.random.random()
                r2 = np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3

        convergence_curve.append(alpha_score)

    return alpha_pos, alpha_score, convergence_curve

def ga_algorithm(obj_function, bounds, population_size=50, generations=100):
    dim = len(bounds)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, bounds[0][0], bounds[0][1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", obj_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    logbook = tools.Logbook()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    result, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, stats=stats,
                                          verbose=False)
    best_individual = tools.selBest(result, 1)[0]

    convergence_curve = [entry['min'] for entry in logbook]

    return best_individual, obj_function(best_individual), convergence_curve, logbook

def woa_algorithm(obj_function, bounds, num_agents=5, max_iter=100):
    dim = len(bounds)

    # Initialize positions of whales
    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))
    leader_pos = np.zeros(dim)
    leader_score = float('inf')

    convergence_curve = []

    for iter in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < leader_score:
                leader_score = fitness
                leader_pos = positions[i, :].copy()

        a = 2 - iter * (2 / max_iter)
        b = 1
        l = np.random.uniform(-1, 1, size=dim)

        for i in range(num_agents):
            r1 = np.random.random()
            r2 = np.random.random()
            A = 2 * a * r1 - a
            C = 2 * r2

            p = np.random.random()
            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * leader_pos - positions[i, :])
                    positions[i, :] = leader_pos - A * D
                else:
                    rand_pos = positions[np.random.randint(0, num_agents), :]
                    D = abs(C * rand_pos - positions[i, :])
                    positions[i, :] = rand_pos - A * D
            else:
                distance_to_leader = abs(leader_pos - positions[i, :])
                positions[i, :] = distance_to_leader * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_pos

        convergence_curve.append(leader_score)

    return leader_pos, leader_score, convergence_curve

def hho_algorithm(obj_function, bounds, num_agents=5, max_iter=100):
    dim = len(bounds)
    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))
    best_position = np.zeros(dim)
    best_score = float('inf')
    convergence_curve = []

    for t in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < best_score:
                best_score = fitness
                best_position = positions[i, :].copy()

        E1 = 2 * (1 - t / max_iter)
        for i in range(num_agents):
            E0 = 2 * np.random.random() - 1
            E = E1 * E0
            q = np.random.random()
            r = np.random.random()

            if abs(E) >= 1:
                rand_index = np.random.randint(0, num_agents)
                X_rand = positions[rand_index, :]
                positions[i, :] = X_rand - r * abs(X_rand - 2 * r * positions[i, :])
            elif q < 0.5:
                if abs(E) < 0.5:
                    positions[i, :] = best_position - E * abs(best_position - positions[i, :])
                else:
                    positions[i, :] = best_position - E * abs(best_position - positions[i, :]) + np.random.randn(dim) * 0.01
            else:
                D_best = abs(best_position - positions[i, :])
                positions[i, :] = D_best * np.exp(E * (np.random.random() - 1)) * np.cos(E * 2 * np.pi) + best_position

        convergence_curve.append(best_score)

    return best_position, best_score, convergence_curve

def geo_algorithm(obj_function, bounds, num_agents=5, max_iter=100):
    dim = len(bounds)
    positions = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_agents, dim))
    best_position = np.zeros(dim)
    best_score = float('inf')
    convergence_curve = []

    for t in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])[0]
            if fitness < best_score:
                best_score = fitness
                best_position = positions[i, :].copy()

        for i in range(num_agents):
            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()
            r4 = np.random.random()
            if r1 < 0.5:
                if r2 < 0.5:
                    positions[i, :] = positions[i, :] + r3 * (best_position - abs(positions[i, :]))
                else:
                    positions[i, :] = best_position - r3 * abs(positions[i, :] - best_position)
            else:
                if r2 < 0.5:
                    positions[i, :] = positions[i, :] + r3 * (best_position - abs(positions[i, :])) + r4 * (positions[i, :] - best_position)
                else:
                    positions[i, :] = positions[i, :] - r3 * (best_position - abs(positions[i, :])) - r4 * (positions[i, :] - best_position)

        convergence_curve.append(best_score)

    return best_position, best_score, convergence_curve

# Optimize PIDA parameters using each optimization algorithm for each benchmark function
optimization_algorithms = [ga_algorithm, gwo_algorithm, woa_algorithm, hho_algorithm, geo_algorithm]
optimization_names = ["GA", "GWO", "WOA", "HHO", "GEO"]

# Store results for plotting
results = {benchmark_name: {optimization_name: None for optimization_name in optimization_names} for benchmark_name in
           benchmark_names}
convergence_curves = {benchmark_name: {optimization_name: None for optimization_name in optimization_names} for
                      benchmark_name in benchmark_names}
stats_logs = {benchmark_name: {optimization_name: None for optimization_name in optimization_names} for benchmark_name
              in benchmark_names}

for benchmark_func, benchmark_name in zip(benchmark_functions, benchmark_names):
    for optimization_algorithm, optimization_name in zip(optimization_algorithms, optimization_names):
        print(f"Optimizing with {optimization_name} using {benchmark_name} Function")

        if optimization_algorithm == ga_algorithm:
            best_params, best_score, convergence_curve, logbook = ga_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
        elif optimization_algorithm == gwo_algorithm:
            best_params, best_score, convergence_curve = gwo_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
            logbook = None  # GWO doesn't provide logbook by default
        elif optimization_algorithm == woa_algorithm:
            best_params, best_score, convergence_curve = woa_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
            logbook = None  # WOA doesn't provide logbook by default
        elif optimization_algorithm == hho_algorithm:
            best_params, best_score, convergence_curve = hho_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
            logbook = None  # HHO doesn't provide logbook by default
        elif optimization_algorithm == geo_algorithm:
            best_params, best_score, convergence_curve = geo_algorithm(
                lambda params: objective_function(params, benchmark_func), bounds)
            logbook = None  # GEO doesn't provide logbook by default

        # Print the optimized parameters
        print(f"Optimized Parameters for {benchmark_name} with {optimization_name}: {best_params}")
        print(f"Best Score for {benchmark_name} with {optimization_name}: {best_score}")

        # Store results for plotting
        results[benchmark_name][optimization_name] = best_params
        convergence_curves[benchmark_name][optimization_name] = convergence_curve
        stats_logs[benchmark_name][optimization_name] = logbook

# Plot results for each benchmark function
for benchmark_name in benchmark_names:
    plt.figure()
    for optimization_name in optimization_names:
        best_params = results[benchmark_name][optimization_name]
        Ka_pida, Kp_pida, Ki_pida, Kd_pida, alpha, beta = best_params

        pida_numerator = [Ka_pida, Kd_pida, Kp_pida, Ki_pida]
        pida_denominator = [1, alpha, beta, 0]
        PIDA = ctrl.TransferFunction(pida_numerator, pida_denominator)

        Amplifier = ctrl.TransferFunction([Ka], [Ta, 1])
        Exciter = ctrl.TransferFunction([Ke], [Te, 1])
        Generator = ctrl.TransferFunction([Kg], [Tg, 1])
        Sensor = ctrl.TransferFunction([Ks], [Ts, 1])

        OpenLoop = PIDA * Amplifier * Exciter * Generator
        ClosedLoop = ctrl.feedback(OpenLoop, Sensor)

        time = np.linspace(0, 10, 1000)
        time, response = ctrl.step_response(ClosedLoop, T=time)

        plt.plot(time, response, label=f"{optimization_name}")

    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.title(f'Closed-Loop Step Response with {benchmark_name} Function')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot convergence curves and stats for each benchmark function
for benchmark_name in benchmark_names:
    plt.figure(figsize=(14, 8))

    # Plot convergence curves
    for optimization_name in optimization_names:
        convergence_curve = convergence_curves[benchmark_name][optimization_name]
        if convergence_curve is not None:
            plt.plot(convergence_curve, label=f"{optimization_name}")
    plt.xlabel('Iterations')
    plt.ylabel('Best Score')
    plt.title(f'Convergence Curve with {benchmark_name} Function')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
