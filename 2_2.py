import numpy as np
import matplotlib.pyplot as plt
import pygad
import random
from deap import base, creator, tools, algorithms


# Example objective function
def objective_function(x):
    return sum(x ** 2)


# Genetic Algorithm (GA)
def genetic_algorithm():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", objective_function)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)

    return hof[0].fitness.values[0]


# Particle Swarm Optimization (PSO)
def particle_swarm_optimization():
    def pso_objective_function(x):
        return objective_function(np.array(x))

    class Particle:
        def __init__(self, x0):
            self.position = np.array(x0)
            self.velocity = np.random.uniform(-1, 1, len(x0))
            self.best_position = self.position.copy()
            self.best_value = pso_objective_function(self.position)
            self.value = self.best_value

        def update_velocity(self, global_best_position, w=0.5, c1=0.8, c2=0.9):
            r1 = np.random.random(len(self.position))
            r2 = np.random.random(len(self.position))
            cognitive_velocity = c1 * r1 * (self.best_position - self.position)
            social_velocity = c2 * r2 * (global_best_position - self.position)
            self.velocity = w * self.velocity + cognitive_velocity + social_velocity

        def update_position(self, bounds):
            self.position += self.velocity
            for i in range(len(self.position)):
                if self.position[i] > bounds[i][1]:
                    self.position[i] = bounds[i][1]
                if self.position[i] < bounds[i][0]:
                    self.position[i] = bounds[i][0]

    def pso_optimize(bounds, num_particles, maxiter):
        dimensions = len(bounds)
        particles = [Particle([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)]) for _ in
                     range(num_particles)]

        global_best_position = particles[0].position.copy()
        global_best_value = pso_objective_function(global_best_position)

        for particle in particles:
            if particle.value < global_best_value:
                global_best_value = particle.value
                global_best_position = particle.position.copy()

        for _ in range(maxiter):
            for particle in particles:
                particle.update_velocity(global_best_position)
                particle.update_position(bounds)
                particle.value = pso_objective_function(particle.position)

                if particle.value < particle.best_value:
                    particle.best_value = particle.value
                    particle.best_position = particle.position.copy()

                if particle.value < global_best_value:
                    global_best_value = particle.value
                    global_best_position = particle.position.copy()

        return global_best_value

    bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    return pso_optimize(bounds, num_particles=30, maxiter=100)


# Grey Wolf Optimization (GWO)
def grey_wolf_optimization():
    def gwo_objective_function(x):
        return objective_function(np.array(x))

    class GreyWolf:
        def __init__(self, x0):
            self.position = np.array(x0)
            self.value = gwo_objective_function(self.position)

        def update_position(self, alpha, beta, delta, a):
            A1, A2, A3 = a * (2 * np.random.random(len(self.position)) - 1), a * (
                        2 * np.random.random(len(self.position)) - 1), a * (
                                     2 * np.random.random(len(self.position)) - 1)
            C1, C2, C3 = 2 * np.random.random(len(self.position)), 2 * np.random.random(
                len(self.position)), 2 * np.random.random(len(self.position))

            D_alpha = abs(C1 * alpha.position - self.position)
            D_beta = abs(C2 * beta.position - self.position)
            D_delta = abs(C3 * delta.position - self.position)

            X1 = alpha.position - A1 * D_alpha
            X2 = beta.position - A2 * D_beta
            X3 = delta.position - A3 * D_delta

            self.position = (X1 + X2 + X3) / 3
            self.value = gwo_objective_function(self.position)

    def gwo_optimize(bounds, num_wolves, maxiter):
        dimensions = len(bounds)
        wolves = [GreyWolf([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)]) for _ in
                  range(num_wolves)]

        alpha = min(wolves, key=lambda wolf: wolf.value)
        beta = min([wolf for wolf in wolves if wolf != alpha], key=lambda wolf: wolf.value)
        delta = min([wolf for wolf in wolves if wolf not in [alpha, beta]], key=lambda wolf: wolf.value)

        a = 2

        for _ in range(maxiter):
            for wolf in wolves:
                wolf.update_position(alpha, beta, delta, a)
            alpha = min(wolves, key=lambda wolf: wolf.value)
            beta = min([wolf for wolf in wolves if wolf != alpha], key=lambda wolf: wolf.value)
            delta = min([wolf for wolf in wolves if wolf not in [alpha, beta]], key=lambda wolf: wolf.value)
            a = 2 - _ * (2 / maxiter)

        return alpha.value

    bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    return gwo_optimize(bounds, num_wolves=30, maxiter=100)


# Run the algorithms and collect final performance values
ga_final_performance = [genetic_algorithm() for _ in range(5)]
pso_final_performance = [particle_swarm_optimization() for _ in range(5)]
gwo_final_performance = [grey_wolf_optimization() for _ in range(5)]

# Calculate mean and standard deviation
ga_mean = np.mean(ga_final_performance)
ga_std = np.std(ga_final_performance)
pso_mean = np.mean(pso_final_performance)
pso_std = np.std(pso_final_performance)
gwo_mean = np.mean(gwo_final_performance)
gwo_std = np.std(gwo_final_performance)

# Print metrics
print(f"GA: Mean = {ga_mean}, Std = {ga_std}")
print(f"PSO: Mean = {pso_mean}, Std = {pso_std}")
print(f"GWO: Mean = {gwo_mean}, Std = {gwo_std}")

# Visualization
algorithms = ['GA', 'PSO', 'GWO']
means = [ga_mean, pso_mean, gwo_mean]
std_devs = [ga_std, pso_std, gwo_std]

plt.figure(figsize=(10, 6))
plt.bar(algorithms, means, yerr=std_devs, capsize=5)
plt.xlabel('Algorithm')
plt.ylabel('Performance Value')
plt.title('Comparison of Final Performance Values')
plt.show()
