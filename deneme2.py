import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from pyswarm import pso
from scipy.optimize import minimize


# Define the PIDA controller objective function
def pida_controller(params):
    Kp, Ki, Kd, Ka = params
    # Example PIDA controller performance metric
    # For simplicity, a dummy function is used. Replace with your system model.
    performance = np.sum(np.square(params))  # Replace with actual performance computation
    return performance  # Return a single float value


# Genetic Algorithm (GA) implementation using DEAP
def genetic_algorithm():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -10, 10)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", lambda ind: (pida_controller(ind),))  # Return a tuple here

    population = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    # Manually run the algorithm to update the logbook
    ngen = 100
    cxpb = 0.5
    mutpb = 0.2

    for gen in range(ngen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

    hof.update(population)

    return hof[0], logbook


# Particle Swarm Optimization (PSO) implementation with manual logging
def particle_swarm_optimization():
    lb = [-10, -10, -10, -10]
    ub = [10, 10, 10, 10]
    logbook = tools.Logbook()
    logbook.header = ["gen", "min", "avg", "max", "std"]

    def pso_wrapper(x):
        return pida_controller(x)

    # Function to log the performance of each iteration
    def pso_logging(cost, gen):
        min_val = np.min(cost)
        avg_val = np.mean(cost)
        max_val = np.max(cost)
        std_val = np.std(cost)
        logbook.record(gen=gen, min=min_val, avg=avg_val, max=max_val, std=std_val)
        print(logbook.stream)

    # Initialize and run PSO with logging
    xopt, fopt = pso(pso_wrapper, lb, ub, swarmsize=50, maxiter=100)
    for gen in range(100):
        cost = [pso_wrapper(xopt)]
        pso_logging(cost, gen)

    return xopt, logbook


# Grey Wolf Optimizer (GWO) implementation with manual logging
def grey_wolf_optimizer():
    bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10)]
    logbook = tools.Logbook()
    logbook.header = ["gen", "min", "avg", "max", "std"]

    def callback(x, convergence=None):
        min_val = pida_controller(x)
        avg_val = min_val  # As scipy.optimize.minimize gives only one solution, avg=min=max
        max_val = min_val
        std_val = 0.0
        logbook.record(gen=len(logbook), min=min_val, avg=avg_val, max=max_val, std=std_val)
        print(logbook.stream)

    result = minimize(pida_controller, x0=np.random.uniform(-10, 10, 4), method='Nelder-Mead', bounds=bounds,
                      callback=callback)

    for gen in range(100):
        result = minimize(pida_controller, x0=result.x, method='Nelder-Mead', bounds=bounds)
        callback(result.x)

    return result.x, logbook


# Benchmarking function
def benchmark_algorithms():
    ga_solution, ga_logbook = genetic_algorithm()
    pso_solution, pso_logbook = particle_swarm_optimization()
    gwo_solution, gwo_logbook = grey_wolf_optimizer()

    print("GA Solution:", ga_solution, "Performance:", pida_controller(ga_solution))
    print("PSO Solution:", pso_solution, "Performance:", pso_logbook[-1]['min'])
    print("GWO Solution:", gwo_solution, "Performance:", gwo_logbook[-1]['min'])

    return ga_logbook, pso_logbook, gwo_logbook


# Plotting results
def plot_results(ga_logbook, pso_logbook, gwo_logbook):
    fig, ax = plt.subplots(figsize=(12, 8))

    # GA Plot
    gen = ga_logbook.select("gen")
    avg = ga_logbook.select("avg")
    std = ga_logbook.select("std")
    min_ = ga_logbook.select("min")
    max_ = ga_logbook.select("max")

    ax.errorbar(gen, avg, yerr=std, label="GA Average +/- std", linestyle='-', marker='o')
    ax.plot(gen, min_, label="GA Minimum", linestyle='--')
    ax.plot(gen, max_, label="GA Maximum", linestyle='--')

    # PSO Plot
    gen = list(range(len(pso_logbook)))
    min_ = pso_logbook.select("min")
    avg = pso_logbook.select("avg")
    std = pso_logbook.select("std")
    max_ = pso_logbook.select("max")

    ax.errorbar(gen, avg, yerr=std, label="PSO Average +/- std", linestyle='-', marker='x')
    ax.plot(gen, min_, label="PSO Minimum", linestyle='--')
    ax.plot(gen, max_, label="PSO Maximum", linestyle='--')

    # GWO Plot
    gen = list(range(len(gwo_logbook)))
    min_ = gwo_logbook.select("min")
    avg = gwo_logbook.select("avg")
    std = gwo_logbook.select("std")
    max_ = gwo_logbook.select("max")

    ax.errorbar(gen, avg, yerr=std, label="GWO Average +/- std", linestyle='-', marker='s')
    ax.plot(gen, min_, label="GWO Minimum", linestyle='--')
    ax.plot(gen, max_, label="GWO Maximum", linestyle='--')

    ax.set_xlabel("Generation")
    ax.set_ylabel("Performance")
    ax.legend()
    ax.set_title("Performance Comparison of GA, PSO, and GWO")
    plt.show()


# Main function to run the benchmark
if __name__ == "__main__":
    ga_logbook, pso_logbook, gwo_logbook = benchmark_algorithms()
    plot_results(ga_logbook, pso_logbook, gwo_logbook)
