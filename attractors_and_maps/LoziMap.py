# Lozi Map is a two-dimensional map similair to the Henon map, but with
# the term -ax^2 replaced by -a*|x|. The Lozi map is a strange attractor.

from pylab import *
import numpy as np


def LoziMap(a, b, x, y):
    return 1 - a * abs(x) + b * y, x


def Normalizer(a, b, x, y):
    LB = -10000.
    UB = 10000.
    if a == 1.5 and -0.5 <= b <= 0.47:
        if b <= 0.4225:
            UB = -0.58728 * b**3 - 0.20915 * b ** 2 - 0.94836 * b - 0.50321
        elif b > 0.4225:
            UB = 0.03116 * b - 0.831737
        if b < -0.37:
            LB = -4.82296 * b ** 2 - 1.47635 * b + 0.4662
        elif -0.37 <= b < 0.4225:
            LB = 0.15227 * b ** 2 + 0.695025 * b + 1.00494
        elif b >= 0.4225:
            LB = -2.55404 * b + 1.30011

    elif 1.8 >= a >= 1.1 and b == 0.1:
        UB = -1.00691 * a + 0.90294
        if a < 1.23703:
            LB = 6.31887 * a**2 - 14.4394 * a + 8.04677
        elif 1.23703 <= a < 1.40334:
            LB = 2.8679 * a**2 - 5.92552 * a + 3.05
        elif 1.40334 <= a:
            LB = -0.05148 * a + 1.14403

    return (x - LB) / (UB - LB), y


def save_data(L, name):
    file = open("./maps/"+name+".dat", 'w')
    for line in L:
        file.write(str(line[0])+"\t"+str(line[1])+"\n")
    file.close()


def main(a, b):
    # Map dependent parameters
    # a = 1.4
    # b = 0.35
    iterations = 50000

    # Initial Condition
    xtemp = np.random.rand()
    ytemp = np.random.rand()

    x = []
    y = []
    x_normalized = []
    y_normalized = []
    x.append(xtemp)
    y.append(ytemp)

    for n in range(iterations):
        xtemp, ytemp = LoziMap(a, b, xtemp, ytemp)
        x.append(xtemp)
        y.append(ytemp)

    for x_i, y_i in zip(x, y):
        x_norm_i, y_norm_i = Normalizer(a, b, x_i, y_i)
        x_normalized.append(x_norm_i)
        y_normalized.append(y_norm_i)

    # Plot the time series
    plot(x_normalized, y_normalized, 'b,')
    show()


if __name__ == '__main__':
    # for ID in range(30):
    #     # TODO change the ID value to have different application for each particle for PSO
    #     appli = rossler_first_return_map(ID)  # appli in [0:1]
    #     save_data(appli, 'rossler_' + str(ID) + '_appli')
    #     appli = lorenz_first_return_map(ID)  # WARNING appli in [0:2]
    #     save_data(appli, 'lorenz_' + str(ID) + '_appli')
    # main(1.4, 0.35)
    main(1.5, 0.45)
