import random
import numpy as np
import operator
import chaos_2018
from deap import base, benchmarks, creator, tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, smin=None, smax=None, best=None)
toolbox = base.Toolbox()
#  number of generations
GEN = 0  # type: int
#  population size
POP_SIZE = 0  # type: int
#  particle dimension size
DIM_SIZE = 0
current_dim = 0
current_part_generate = 0
current_particle = 0
current_gen = 0
Map1 = []
Map2 = []


# Populate the current experiment's maps
def generate_maps(generation_size, population_size, dimension_size, ID_start_point):
    global Map1, Map2
    ID = ID_start_point
    total_number_of_maps = dimension_size * population_size
    for particle in range(population_size):
        Map1_layer = np.random.rand(generation_size, 0)
        Map2_layer = np.random.rand(generation_size, 0)
        for _ in range(dimension_size):
            Map1_layer = np.c_[Map1_layer, chaos_2018.rossler_first_return_map(ID, generation_size)]
            ID += 1
            Map2_layer = np.c_[Map2_layer, chaos_2018.rossler_first_return_map(ID, generation_size)]
            ID += 1
        Map1.append(Map1_layer)
        Map2.append(Map2_layer)


def save_data(file_name, average_mins, problem_type):
    file = open("output/" + problem_type+"/"+ file_name + ".dat", 'w')
    for record in average_mins:
        file.write(str(record) + "\n")
    file.close()


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best, phi1, phi2):
    global current_particle, current_gen

    # get dimension values for specific current particle in MAP1,2
    temp_chaos1 = Map1[current_particle][current_gen,:]
    temp_chaos2 = Map2[current_particle][current_gen,:]

    # assign them as the new random variables
    u1 = (temp_chaos1[i]*phi1 for i in range(len(part)))
    u2 = (temp_chaos2[i]*phi2 for i in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part[:] = list(map(operator.add, part, part.speed))

    #  Increase particle index
    current_particle += 1


def main():
    global current_gen, Map1, Map2, current_particle, Normal_Map1, Normal_Map2
    pop = toolbox.population(n=POP_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None

    current_gen = 0
    for g in range(GEN):
        current_particle = 0

        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        current_gen += 1

    return pop, logbook, best


def rossler_cluster_run(generation, particle, dimension, experiment, problem_type, PHI1, PHI2, SMIN, SMAX, PMIN, PMAX):
    global GEN, POP_SIZE, EXPERIMENT, DIM_SIZE, toolbox, current_dim, current_part_generate
    print("RosslerPSO has started solving " + problem_type.swapcase() + " problem with number of generations: "+str(generation)+", population size: "+str(particle)+", particle dimension: "+str(dimension)+" with experiment size of "+str(experiment))
    GEN = generation
    POP_SIZE = particle
    EXPERIMENT = experiment
    DIM_SIZE = dimension

    # Register parameters
    toolbox.register("particle", generate, size=DIM_SIZE, pmin=PMIN, pmax=PMAX, smin=SMIN, smax=SMAX)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=PHI1, phi2=PHI2)
    # Register selected problem
    if problem_type == "sphere":
        toolbox.register("evaluate", benchmarks.sphere)
    elif problem_type == "griewank":
        toolbox.register("evaluate", benchmarks.griewank)
    elif problem_type == "rastrigin":
        toolbox.register("evaluate", benchmarks.rastrigin)
    elif problem_type == "schaffer":
        toolbox.register("evaluate", benchmarks.schaffer)
    elif problem_type == "rosenbrock":
        toolbox.register("evaluate", benchmarks.rosenbrock)
    elif problem_type == "schwefel":
        toolbox.register("evaluate", benchmarks.schwefel)
    elif problem_type == "ackley":
        toolbox.register("evaluate", benchmarks.ackley)
    elif problem_type == "himmelblau":
        toolbox.register("evaluate", benchmarks.himmelblau)

    # number of experiments
    average_mins = [0.] * GEN
    for r in range(EXPERIMENT):

        # GENERATE Lorentz Chaotic Number Maps in Map1 and Map2 global variables
        ID_start_point = r * particle * dimension * 2
        generate_maps(generation, particle, dimension, ID_start_point)

        pop, logbook, best = main()
        print(logbook.select("min"))
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        for i in range(GEN):
            average_mins[i] += fit_mins[i] / EXPERIMENT

    file_name = "rossler_results"
    save_data(file_name, average_mins, problem_type)
    # plt.title(problem_type.swapcase())
    # plt.xlabel("Generation")
    # plt.ylabel("Minimum Fitness")
    # plt.plot(gen, average_mins)
    # plt.show()


if __name__ == '__main__':
    rossler_cluster_run(100, 5, 10, 2, "sphere")

