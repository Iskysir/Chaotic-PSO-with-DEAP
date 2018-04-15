import numpy as np
import matplotlib.pyplot as plt


def main():
    file_names = ["basic", "logistic", "lozi", "lorenz"]
    colors = ['r-', 'g-', 'b-', 'y-']
    data = [None] * len(file_names)
    gen = np.arange(200)
    for i in range(len(file_names)):
        data[i] = np.genfromtxt("output/" + file_names[i] + "_results.dat")
        plt.plot(gen, data[i], colors[i])

    plt.xlabel("Generation")
    plt.ylabel("Minimum Fitness")
    plt.legend([file_names[0], file_names[1], file_names[2], file_names[3]], loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
