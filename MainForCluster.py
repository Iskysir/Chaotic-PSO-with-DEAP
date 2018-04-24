import LorenzPSO
import BasicPSO
import LogisticPSO
import RosslerPSO
import LoziPSO


def main(generation, particle, dimension, experiment, problem_type, phi1, phi2, smin, smax, pmin, pmax):
    BasicPSO.basic_cluster_run(generation, particle, dimension, experiment, problem_type, phi1, phi2, smin, smax, pmin, pmax)
    LogisticPSO.logistic_cluster_run(generation, particle, dimension, experiment, problem_type, phi1, phi2, smin, smax, pmin, pmax)
    LoziPSO.lozi_cluster_run(generation, particle, dimension, experiment, problem_type, phi1, phi2, smin, smax, pmin, pmax)
    LorenzPSO.lorenz_cluster_run(generation, particle, dimension, experiment, problem_type, phi1, phi2, smin, smax, pmin, pmax)
    RosslerPSO.rossler_cluster_run(generation, particle, dimension, experiment, problem_type, phi1, phi2, smin, smax, pmin, pmax)


if __name__ == '__main__':
    gen = 100
    part = 5
    dim = 5
    exp = 1
    # Acceleration constants
    phi1 = 2.0
    phi2 = 2.0
    # Speed limits
    smin = -0.5
    smax = 0.5
    # Position limits
    pmin = -100
    pmax = 100
    problem_types = ["sphere", "griewank", "rosenbrock", "schaffer", "rastrigin", "schwefel", "ackley", "himmelblau"]
    for problem_type in problem_types:
        # Set initial position limits
        if problem_type == "sphere" or problem_type == "griewank":
            pmin, pmax = -5.12, 5.12
            
        elif problem_type == "schaffer":
            pmin, pmax = -100., 100.
            
        elif problem_type == "rastrigin":
            pmin, pmax = -600., 600.
            
        elif problem_type == "rosenbrock":
            pmin, pmax = -30., 30.
            
        elif problem_type == "schwefel":
            pmin, pmax = -500., 500.
            
        elif problem_type == "ackley":
            pmin, pmax = -15., 30.
            
        elif problem_type == "himmelblau":
            pmin, pmax = -6., 6.
            
        print("Solving " + problem_type.swapcase() + " Problem")
        main(gen, part, dim, exp, problem_type, phi1, phi2, smin, smax, pmin, pmax)
    # main(gen, part, dim, exp, problem_types[0], phi1, phi2, smin, smax, pmin, pmax)
    # main(gen, part, dim, exp, problem_types[1], phi1, phi2, smin, smax, pmin, pmax)
    # main(gen, part, dim, exp, problem_types[2], phi1, phi2, smin, smax, pmin, pmax)
    # main(gen, part, dim, exp, problem_types[3], phi1, phi2, smin, smax, pmin, pmax)
    # main(gen, part, dim, exp, problem_types[4], phi1, phi2, smin, smax, pmin, pmax)
    # main(gen, part, dim, exp, problem_types[5], phi1, phi2, smin, smax, pmin, pmax)
    # main(gen, part, dim, exp, problem_types[6], phi1, phi2, smin, smax, pmin, pmax)
    # main(gen, part, dim, exp, problem_types[7], phi1, phi2, smin, smax, pmin, pmax)