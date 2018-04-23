import operator
import random
import numpy
import matplotlib.pyplot as plt

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, smin=None, smax=None, best=None)
toolbox = base.Toolbox()

#  number of generations
GEN = 0  # type: int
#  population size
POP_SIZE = 0  # type: int
# number of dimensions in a particle
DIM_SIZE = 0  # type: int
# # type of the problem
# problem_type = str
# # range min
# problem_smin = float
# # range max
# problem_smax = float

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
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part[:] = list(map(operator.add, part, part.speed))


def main():
    pop = toolbox.population(n=POP_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None

    for g in range(GEN):
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

    return pop, logbook, best


# CALL THIS FUNCTION FROM OUTSIDE OF THE SCRIPT
def basic_cluster_run(generation, particle, dimension, experiment, problem_type, PHI1, PHI2, SMIN, SMAX, PMIN, PMAX):
    global GEN, POP_SIZE, EXPERIMENT, DIM_SIZE, toolbox
    print("BasicPSO has started solving " + problem_type.swapcase() + " problem with number of generations: "+str(generation)+", population size: "+str(particle)+", particle dimension: "+str(dimension)+" with experiment size of "+str(experiment))
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


    average_mins = [0.] * GEN
    for r in range(EXPERIMENT):
        pop, logbook, best = main()
        print(logbook.select("min"))
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        for i in range(GEN):
            average_mins[i] += fit_mins[i] / EXPERIMENT

    file_name = "basic_results"
    #save_data(file_name, average_mins, problem_type)

    # plt.title(problem_type.swapcase())
    # plt.xlabel("Generation")
    # plt.ylabel("Minimum Fitness")
    # plt.plot(gen, average_mins)
    # plt.show()


if __name__ == "__main__":
    basic_cluster_run(200, 20, 100, 30, "rastrigin")

