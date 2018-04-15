import random
from typing import Any, Union, Generator, List, Tuple
import numpy
import operator
import matplotlib.pyplot as plt
import chaos_2018
from deap import base, benchmarks, creator, tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, smin=None, smax=None, best=None)

#  number of generations
GEN = 200  # type: int
#  population size
SIZE = 10  # type: int
pos0 = []  # type: List[Double]
vel0 = []  # type: List[Double]


def LoziMap(x, y):
    # Map dependent parameters
    a = 1.7
    b = 0.3
    return 1. - a * abs(x) + b * y, x


def save_data(file_name, average_mins):
    file = open("output/" + file_name + ".dat", 'w')
    for record in average_mins:
        file.write(str(record) + "\n")
    file.close()


def generate(size, pmin, pmax, smin, smax):
    global pos0
    for _ in range(size):
        pos0.append(random.uniform(0, 1.))
    for _ in range(size):
        vel0.append(random.uniform(0, 1.))
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    # part = creator.Particle(pos0[i] for i in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best, phi1, phi2):
    global pos0, vel0
    posN = []
    velN = []
    for p, v in zip(pos0, vel0):
        pNext, vNext = LoziMap(p, v)
        posN.append(pNext )
        velN.append(vNext )
    u1 = (posN[i] for i in range(len(part)))
    u2 = (velN[i] for i in range(len(part)))
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
    pos0 = posN
    vel0 = velN


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=60, pmin=-5.12, pmax=5.12, smin=-0.5, smax=0.5)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=1.0, phi2=1.0)
toolbox.register("evaluate", benchmarks.sphere)


def main():
    pop = toolbox.population(n=SIZE)
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


if __name__ == '__main__':
    noe = 30  # number of experiments
    average_mins = [0.] * GEN
    for r in range(noe):
        pop, logbook, best = main()
        print(logbook.select("min"))
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        for i in range(GEN):
            average_mins[i] += fit_mins[i]/noe

    file_name = "lozi_results"
    save_data(file_name, average_mins)

    plt.xlabel("Generation")
    plt.ylabel("Minimum Fitness")
    plt.plot(gen, average_mins)
    plt.show()
