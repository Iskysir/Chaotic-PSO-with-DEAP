import random
from typing import Any, Union, Generator
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
POP_SIZE = 10  # type: int
pos0 = []  # type: List[Double]
vel0 = []  # type: List[Double]

data = numpy.genfromtxt("maps/lorenz_" + str(0) + "_appli.dat")
# dataVEL = numpy.genfromtxt("maps/lorenz_" + str(1) + "_appli.dat")
count = 0


def LorenzMap(count):
    # X = data[:, 0]
    # Y = data[:, 1]
    return data[count, 0], data[count, 1]


def save_data(file_name, average_mins):
    file = open("output/" + file_name + ".dat", 'w')
    for record in average_mins:
        file.write(str(record) + "\n")
    file.close()


def generate(size, pmin, pmax, smin, smax):
    global pos0, count
    count = 0
    for _ in range(size):
        pos0.append(data[count, 0])
        vel0.append(data[count, 1])
        count += 1
        print(count)
    print("Generation Complete")
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    # part = creator.Particle(pos0[i] for i in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best, phi1, phi2):
    global pos0, vel0, count
    posN = []
    velN = []
    for p, v in zip(pos0, vel0):
        pNext, vNext = LorenzMap(count)
        posN.append(pNext)
        velN.append(vNext)
        count += 1
        print(count)
    print("Update Complete")
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
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", benchmarks.sphere)


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


if __name__ == '__main__':
    noe = 1  # number of experiments
    average_mins = [0.] * GEN
    for r in range(noe):
        ID = r
        data = numpy.genfromtxt("maps/lorenz_" + str(ID) + "_appli.dat")
        pop, logbook, best = main()
        print(logbook.select("min"))
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        for i in range(GEN):
            average_mins[i] += fit_mins[i]/noe

    file_name = "lorenz_results"
    save_data(file_name, average_mins)

    plt.xlabel("Generation")
    plt.ylabel("Minimum Fitness")
    plt.plot(gen, average_mins)
    plt.show()
