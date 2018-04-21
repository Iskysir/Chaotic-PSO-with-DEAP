import BasicPSO
import LogisticPSO
import LoziPSO
import LorenzPSO
import RosslerPSO


def main(generation, particle, dimension, experiment):
    BasicPSO.basic_cluster_run(generation, particle, dimension, experiment)
    LogisticPSO.logistic_cluster_run(generation, particle, dimension, experiment)
    LoziPSO.lozi_cluster_run(generation, particle, dimension, experiment)
    LorenzPSO.lorenz_cluster_run(generation, particle, dimension, experiment)
    RosslerPSO.rossler_cluster_run(generation, particle, dimension, experiment)


if __name__ == '__main__':
    gen = 300
    part = 50
    dim = 150
    exp = 30
    main(gen, part, dim, exp)

