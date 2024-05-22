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


# Particle Swarm Optimization (PSO) implementation using pyswarm
def particle_swarm_optimization():
    lb = [-10, -10, -10, -10]
    ub = [10, 10, 10, 10]
    xopt, fopt = pso(pida_controller, lb, ub, swarmsize=50, maxiter=100)
    return xopt, fopt


# Grey Wolf Optimizer (GWO) implementation using scipy.optimize.minimize with Nelder-Mead (as a placeholder)
def grey_wolf_optimizer():
    bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10)]
    result = minimize(pida_controller, x0=np.random.uniform(-10, 10, 4), method='Nelder-Mead', bounds=bounds)
    return result.x, result.fun


# Benchmarking function
def benchmark_algorithms():
    ga_solution, ga_logbook = genetic_algorithm()
    pso_solution, pso_performance = particle_swarm_optimization()
    gwo_solution, gwo_performance = grey_wolf_optimizer()

    print("GA Solution:", ga_solution, "Performance:", pida_controller(ga_solution))
    print("PSO Solution:", pso_solution, "Performance:", pso_performance)
    print("GWO Solution:", gwo_solution, "Performance:", gwo_performance)

    return ga_logbook


# Plotting results
def plot_results(logbook):
    gen = logbook.select("gen")
    avg = logbook.select("avg")
    std = logbook.select("std")
    min_ = logbook.select("min")
    max_ = logbook.select("max")

    plt.figure()
    plt.errorbar(gen, avg, yerr=std, label="Average +/- std")
    plt.plot(gen, min_, label="Minimum")
    plt.plot(gen, max_, label="Maximum")
    plt.xlabel("Generation")
    plt.ylabel("Performance")
    plt.legend()
    plt.show()


# Main function to run the benchmark
if __name__ == "__main__":
    ga_logbook = benchmark_algorithms()
    plot_results(ga_logbook)
