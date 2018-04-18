# Lozi Map is a two-dimensional map similair to the Henon map, but with
# the term -ax^2 replaced by -a*|x|. The Lozi map is a strange attractor.

from pylab import *
from numpy import random


def LoziMap(a, b, x, y):
	temp = x
	x = 1 - a * abs(x) + b * y
	y = temp
	if not (0 < x < 1 and 0 < y < 1):
		LoziMap(a, b, x, y)
	return x, y


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
	xtemp = random.uniform(0., 1.)
	ytemp = random.uniform(0., 1.)

	x = [max(xtemp, -xtemp)]
	y = [max(ytemp, -ytemp)]

	for n in range(iterations):
		xtemp, ytemp = LoziMap(a, b, xtemp, ytemp)
		x.append(xtemp)
		y.append(ytemp)

	# Plot the time series
	plot(x, y, 'b,')
	show()


if __name__ == '__main__':
    # for ID in range(30):
    #     # TODO change the ID value to have different application for each particle for PSO
    #     appli = rossler_first_return_map(ID)  # appli in [0:1]
    #     save_data(appli, 'rossler_' + str(ID) + '_appli')
    #     appli = lorenz_first_return_map(ID)  # WARNING appli in [0:2]
    #     save_data(appli, 'lorenz_' + str(ID) + '_appli')
	# main(1.4, 0.35)
	main(1.7, 0.45)

