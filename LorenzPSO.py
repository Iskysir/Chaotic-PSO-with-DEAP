import random
import numpy as np
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
#  particle dimension size
DIM_SIZE = 0
current_dim = 0
current_part_generate = 0
current_particle = 0
current_gen = 0
Map1 = []
Map2 = []
Normal_Map1 = []
Normal_Map2 = []
# For normalize x or y or z set 0 or 1 or 2
norm_var_index = 1
# Define arbitrary values to prevent error
x_UB = 22
x_LB = -21.5
y_UB = 29.5
y_LB = -29
z_UB = 54.5
z_LB = 0


def BoundEstimator():
    global x_LB, x_UB, y_LB, y_UB, z_LB, z_UB
    xs_list = []
    ys_list = []
    zs_list = []
    for _ in range(100):
        dt = 0.01
        stepCnt = GEN

        # Need one more for the initial values
        xs = np.empty((stepCnt + 1,))
        ys = np.empty((stepCnt + 1,))
        zs = np.empty((stepCnt + 1,))

        # Setting initial values
        xs[0], ys[0], zs[0] = (np.random.rand(), np.random.rand(), np.random.rand())

        # Stepping through "time".
        for i in range(stepCnt):
            # Derivatives of the X, Y, Z state
            xs[i + 1], ys[i + 1], zs[i + 1] = lorenz(xs[i], ys[i], zs[i])
        xs_list.extend(xs)
        ys_list.extend(ys)
        zs_list.extend(zs)
    x_UB = max(xs_list)
    x_LB = min(xs_list)
    y_UB = max(ys_list)
    y_LB = min(ys_list)
    z_UB = max(zs_list)
    z_LB = min(zs_list)
    print("x: Upperbound and Lowerbound")
    print(x_UB, x_LB)
    print("y: Upperbound and Lowerbound")
    print(y_UB, y_LB)
    print("z: Upperbound and Lowerbound")
    print(z_UB, z_LB)
    return


def lorenz(x, y, z):
    dt = 0.01
    s = 10.
    r = 28.
    b = 8./3.
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x + x_dot * dt, y + y_dot * dt, z + z_dot * dt


# Normalized Algorithm based on paper
def normalizer(X, Y, Z):
    return (X - x_LB) / (x_UB - x_LB), (Y - y_LB) / (y_UB - y_LB), (Z - z_LB) / (z_UB - z_LB)


#  Change each element inside the MAP to next chaotic value
def chaoticFunc(MAP):
    for i in range(POP_SIZE):
        for j in range(DIM_SIZE):
            MAP[0][i][j], MAP[1][i][j], MAP[2][i][j] = lorenz(MAP[0][i][j], MAP[1][i][j], MAP[2][i][j])
    return MAP


def save_data(file_name, average_mins):
    file = open("output/" + file_name + ".dat", 'w')
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
    global current_particle
    # get dimension values for specific current particle in MAP1,2
    temp_chaos1 = Normal_Map1[norm_var_index][current_particle][:]
    temp_chaos2 = Normal_Map2[norm_var_index][current_particle][:]
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
    # map[:, i] = appli
    # CHAOTIC MAP GENERATOR
    # MAP = np.ndarray(shape=(POP_SIZE, DIM_SIZE), dtype=float, order='F')
    # Initial MAP contains all randoms with PARTxDIM
    Map1.append([[random.uniform(0, 1.) for _ in range(DIM_SIZE)] for _ in range(POP_SIZE)])
    Map1.append([[random.uniform(0, 1.) for _ in range(DIM_SIZE)] for _ in range(POP_SIZE)])
    Map1.append([[random.uniform(0, 1.) for _ in range(DIM_SIZE)] for _ in range(POP_SIZE)])
    Map2.append([[random.uniform(0, 1.) for _ in range(DIM_SIZE)] for _ in range(POP_SIZE)])
    Map2.append([[random.uniform(0, 1.) for _ in range(DIM_SIZE)] for _ in range(POP_SIZE)])
    Map2.append([[random.uniform(0, 1.) for _ in range(DIM_SIZE)] for _ in range(POP_SIZE)])

    Normal_Map1 = Map1
    Normal_Map2 = Map2
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

        # Update maps with next chaotic level
        Map1 = chaoticFunc(Map1)
        Map2 = chaoticFunc(Map2)

        # Normalize MAPS
        for i in range(POP_SIZE):
            for j in range(DIM_SIZE):
                Normal_Map1[0][i][j], Normal_Map1[1][i][j], Normal_Map1[2][i][j] = normalizer(Map1[0][i][j], Map1[1][i][j], Map1[2][i][j])
                Normal_Map2[0][i][j], Normal_Map2[1][i][j], Normal_Map2[2][i][j] = normalizer(Map2[0][i][j], Map2[1][i][j], Map2[2][i][j])
        # Normalize MAPS end
        current_gen += 1

    return pop, logbook, best


def lorenz_cluster_run(generation, particle, dimension, experiment):
    global GEN, POP_SIZE, EXPERIMENT, DIM_SIZE, toolbox, current_dim, current_part_generate
    print("Lorenz PSO algorithm has started with number of generations: "+str(generation)+", population size: "+str(particle)+", particle dimension: "+str(dimension)+" with experiment size of "+str(experiment))
    GEN = generation
    POP_SIZE = particle
    EXPERIMENT = experiment
    DIM_SIZE = dimension

    toolbox.register("particle", generate, size=DIM_SIZE, pmin=-5.12, pmax=5.12, smin=-0.5, smax=0.5)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
    toolbox.register("evaluate", benchmarks.sphere)

    # Set normalizer bound values at first
    BoundEstimator()
    # number of experiments
    average_mins = [0.] * GEN
    for r in range(EXPERIMENT):
        pop, logbook, best = main()
        print(logbook.select("min"))
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        for i in range(GEN):
            average_mins[i] += fit_mins[i] / EXPERIMENT

    file_name = "lorenz_results"
    save_data(file_name, average_mins)

    # plt.xlabel("Generation")
    # plt.ylabel("Minimum Fitness")
    # plt.plot(gen, average_mins)
    # plt.show()

    del toolbox, pop, logbook, best


if __name__ == '__main__':
    lorenz_cluster_run(500, 20, 60, 5)
