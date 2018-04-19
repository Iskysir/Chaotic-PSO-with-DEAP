import random
import numpy
import operator
import matplotlib.pyplot as plt

from deap import base, benchmarks, creator, tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, smin=None, smax=None, best=None)
toolbox = base.Toolbox()
#  number of generations
GEN = 0  # type: int
#  population size
POP_SIZE = 0  # type: int
#  particle DIMENSION
DIMENSION = 0
chaos1 = []
chaos2 = []
MAP = []


def LogisticMap(xn):
    r = 4.  # r is the chaos growth rate
    return r * xn * (1. - xn)


#  Select which chaotic map to use
def chaoticFunc(xn):
    return LogisticMap(xn)


def save_data(file_name, average_mins):
    file = open("output/" + file_name + ".dat", 'w')
    for record in average_mins:
        file.write(str(record) + "\n")
    file.close()


def generate(size, pmin, pmax, smin, smax):
    global chaos1, chaos2
    for _ in range(size):
        chaos1.append(random.uniform(0, 1.))
    for _ in range(size):
        chaos2.append(random.uniform(0, 1.))

    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best, phi1, phi2):
    global chaos1, chaos2
    posN = []
    posM = []
    for val in chaos1:
        posN.append(chaoticFunc(val))
    for val in chaos2:
        posM.append(chaoticFunc(val))
    u1 = (posN[i] for i in range(len(part)))
    u2 = (posM[i] for i in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part[:] = list(map(operator.add, part, part.speed))
    #  Update pos0 with new elements of chaos in the end
    chaos1 = posN
    chaos2 = posM


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


def logistic_cluster_run(generation, particle, dimension, experiment):
    global GEN, POP_SIZE, EXPERIMENT, DIMENSION, toolbox
    GEN = generation
    POP_SIZE = particle
    EXPERIMENT = experiment
    DIMENSION = dimension
    toolbox.register("particle", generate, size=DIMENSION, pmin=-5.12, pmax=5.12, smin=-0.5, smax=0.5)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=1.0, phi2=1.0)
    toolbox.register("evaluate", benchmarks.sphere)

    # number of experiments
    average_mins = [0.] * GEN
    for r in range(EXPERIMENT):
        pop, logbook, best = main()
        print(logbook.select("min"))
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        for i in range(GEN):
            average_mins[i] += fit_mins[i] / EXPERIMENT

    file_name = "logistic_results"
    save_data(file_name, average_mins)

    plt.xlabel("Generation")
    plt.ylabel("Minimum Fitness")
    plt.plot(gen, average_mins)
    plt.show()


if __name__ == '__main__':
    logistic_cluster_run(100, 10, 5, 30)
