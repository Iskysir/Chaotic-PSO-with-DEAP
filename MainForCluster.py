from pso_algorithms import RosslerPSO, LoziPSO, BasicPSO, LogisticPSO, LorenzPSO


def main(generation, particle, dimension, experiment, problem_type):
    BasicPSO.basic_cluster_run(generation, particle, dimension, experiment, problem_type)
    LogisticPSO.logistic_cluster_run(generation, particle, dimension, experiment, problem_type)
    LoziPSO.lozi_cluster_run(generation, particle, dimension, experiment, problem_type)
    LorenzPSO.lorenz_cluster_run(generation, particle, dimension, experiment, problem_type)
    RosslerPSO.rossler_cluster_run(generation, particle, dimension, experiment, problem_type)


if __name__ == '__main__':
    gen = 500
    part = 20
    dim = 150
    exp = 30
    problem_types = ["sphere", "griewank", "rosenbrock", "schaffer", "rastrigin", "schwefel", "ackley", "himmelblau"]
    # for problem_type in problem_types:
    #     print("Solving " + problem_type.swapcase() + " Problem")
    #     main(gen, part, dim, exp, problem_type)
    # main(gen, part, dim, exp, problem_types[0])
    # main(gen, part, dim, exp, problem_types[1])
    # main(gen, part, dim, exp, problem_types[2])
    # main(gen, part, dim, exp, problem_types[3])
    # main(gen, part, dim, exp, problem_types[4])
    main(gen, part, dim, exp, problem_types[5])
    main(gen, part, dim, exp, problem_types[6])
    main(gen, part, dim, exp, problem_types[7])