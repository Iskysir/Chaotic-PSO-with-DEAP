import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


#style.use("fivethirtyeight")
def main():
    # folder_names = ["sphere", "griewank", "rosenbrock", "schaffer", "rastrigin", "schwefel", "ackley", "himmelblau"]
    folder_names = ["rosenbrock"]
    file_names = ["basic", "logistic", "lozi", "lorenz", "rossler"]
#    colors = ['r-', 'g-', 'b-', 'y-', '-']
    data = [None] * len(file_names)

    # fig = plt.figure()
    # ax_list = [fig.add_subplot(231),
    #            fig.add_subplot(232),
    #            fig.add_subplot(233),
    #            fig.add_subplot(234),
    #            fig.add_subplot(235)]

    for folder in folder_names:
        for i in range(len(file_names)):
            data[i] = np.genfromtxt("output/" + folder + "/" + file_names[i] + "_results.dat")
            gen = np.arange(len(data[i]))
            #ax_list[folder_names.index(folder)].plot(gen, data[i], '-')
            plt.plot(gen, data[i], '-')
        plt.title(folder.swapcase())
        plt.xlabel("Generation")
        plt.ylabel("Minimum Fitness")
        plt.legend([file_names[0], file_names[1], file_names[2], file_names[3], file_names[4]], loc='upper right')
        plt.savefig('/Users/tolgasaglik/PycharmProjects/cPSO/plots/' + folder + '_plot.png', dpi=500)
        plt.show()
        plt.clf()


if __name__ == '__main__':
    main()
