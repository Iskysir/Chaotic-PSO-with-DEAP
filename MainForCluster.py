import BasicPSO


def main(generation, particle, dimension, experiment):
    BasicPSO.basic_cluster_run(generation, particle, dimension, experiment)
    # LogisticPSO.logistic_cluster_run(generation, particle, dimension, experiment)
    # LoziPSO.lozi_cluster_run(generation, particle, dimension, experiment)
    # RosslerPSO.rossler_cluster_run(generation, particle, dimension, experiment)
    # LorenzPSO.lorenz_cluster_run(generation, particle, dimension, experiment)


if __name__ == '__main__':
    gen = 200
    part = 20
    dim = 60
    exp = 5
    main(gen, part, dim, exp)
    # DataPlotter.main

