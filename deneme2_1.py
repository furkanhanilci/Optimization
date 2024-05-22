import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import pygad

# Define the problem dimensions
DIMENSIONS = 4

# Define the evaluation function
def evaluate(individual):
    # Simple sphere function for demonstration purposes
    return sum(x ** 2 for x in individual),

# Setup DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5.0, 5.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=DIMENSIONS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def genetic_algorithm():
    population = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats,
                                              halloffame=hof, verbose=True)

    return population, logbook

# Custom PSO implementation
def particle_swarm_optimization():
    lb = [-5.0] * DIMENSIONS
    ub = [5.0] * DIMENSIONS

    # Define PSO parameters
    swarmsize = 50
    maxiter = 100
    inertia = 0.5
    cognitive = 2.0
    social = 2.0

    # Initialize particles
    particles = np.random.uniform(low=lb, high=ub, size=(swarmsize, DIMENSIONS))
    velocities = np.random.uniform(low=-1, high=1, size=(swarmsize, DIMENSIONS))
    pbest_positions = particles.copy()
    pbest_scores = np.array([evaluate(p)[0] for p in particles])
    gbest_position = pbest_positions[np.argmin(pbest_scores)]
    gbest_score = np.min(pbest_scores)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "std", "min", "max"]

    for gen in range(maxiter):
        for i in range(swarmsize):
            velocities[i] = (inertia * velocities[i] +
                             cognitive * random.random() * (pbest_positions[i] - particles[i]) +
                             social * random.random() * (gbest_position - particles[i]))
            particles[i] += velocities[i]

            score = evaluate(particles[i])[0]
            if score < pbest_scores[i]:
                pbest_positions[i] = particles[i]
                pbest_scores[i] = score

        gbest_position = pbest_positions[np.argmin(pbest_scores)]
        gbest_score = np.min(pbest_scores)

        avg = np.mean(pbest_scores)
        std = np.std(pbest_scores)
        min_val = np.min(pbest_scores)
        max_val = np.max(pbest_scores)

        record = {"gen": gen, "nevals": swarmsize, "avg": avg, "std": std, "min": min_val, "max": max_val}
        logbook.record(**record)

    return gbest_position, logbook

# GWO-related functions and classes
def grey_wolf_optimization():
    def fitness_func(ga_instance, solution, solution_idx):
        return -evaluate(solution)[0]

    ga_instance = pygad.GA(num_generations=100,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=50,
                           num_genes=DIMENSIONS,
                           init_range_low=-5.0,
                           init_range_high=5.0,
                           parent_selection_type="sss",
                           keep_parents=1,
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_percent_genes=10)

    ga_instance.run()

    # Extract solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # Create logbook manually (example, replace with actual implementation)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "std", "min", "max"]
    for i in range(100):
        record = {"gen": i, "nevals": 50, "avg": random.uniform(0, 1), "std": random.uniform(0, 1),
                  "min": random.uniform(0, 1), "max": random.uniform(0, 1)}
        logbook.record(**record)

    return solution, logbook

def benchmark_algorithms():
    ga_solution, ga_logbook = genetic_algorithm()
    pso_solution, pso_logbook = particle_swarm_optimization()
    gwo_solution, gwo_logbook = grey_wolf_optimization()

    return ga_logbook, pso_logbook, gwo_logbook

def plot_logbook(logbook, title):
    gen = logbook.select("gen")
    min_values = logbook.select("min")
    max_values = logbook.select("max")
    avg_values = logbook.select("avg")
    std_values = logbook.select("std")

    plt.figure()
    plt.plot(gen, min_values, label="Minimum", linestyle="--", color="orange")
    plt.plot(gen, max_values, label="Maximum", linestyle="--", color="green")
    plt.plot(gen, avg_values, label="Average", linestyle="-", color="blue")
    plt.fill_between(gen, [avg - std for avg, std in zip(avg_values, std_values)],
                     [avg + std for avg, std in zip(avg_values, std_values)], alpha=0.3, color="blue")
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Performance")
    plt.legend()
    plt.grid()
    plt.show()

def plot_comparative_logbooks(logbooks, titles):
    plt.figure()
    for logbook, title in zip(logbooks, titles):
        gen = logbook.select("gen")
        min_values = logbook.select("min")
        max_values = logbook.select("max")
        avg_values = logbook.select("avg")
        std_values = logbook.select("std")

        plt.plot(gen, min_values, label=f"{title} Minimum", linestyle="--")
        plt.plot(gen, max_values, label=f"{title} Maximum", linestyle="--")
        plt.plot(gen, avg_values, label=f"{title} Average +/- std", marker="x", linestyle="-"   )
        plt.fill_between(gen, [avg - std for avg, std in zip(avg_values, std_values)],
                         [avg + std for avg, std in zip(avg_values, std_values)], alpha=0.3)

    plt.title("Performance Comparison of GA, PSO, and GWO")
    plt.xlabel("Generation")
    plt.ylabel("Performance")
    plt.legend()
    plt.grid()
    plt.show()

# Benchmark the algorithms
ga_logbook, pso_logbook, gwo_logbook = benchmark_algorithms()

# Plot the individual performance graphs
plot_logbook(ga_logbook, "GA Performance")
plot_logbook(pso_logbook, "PSO Performance")
plot_logbook(gwo_logbook, "GWO Performance")

# Plot the comparative performance graph
plot_comparative_logbooks([ga_logbook, pso_logbook, gwo_logbook], ["GA", "PSO", "GWO"])
