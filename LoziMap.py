# Lozi Map is a two-dimensional map similair to the Henon map, but with
# the term -ax^2 replaced by -a*|x|. The Lozi map is a strange attractor.

from pylab import *
from numpy import random


def LoziMap(a, b, x, y):
	return 1 - a*abs(x) + y, b*x


def main(a, b):
	# Map dependent parameters
	# a = 1.4
	# b = 0.35
	iterations = 100000

	# Initial Condition
	xtemp = random.uniform(0, 1.0)
	ytemp = random.uniform(0, 1.0)

	x = [xtemp]
	y = [ytemp]

	for n in range(0, iterations):
		xtemp, ytemp = LoziMap(a, b, xtemp, ytemp)
		x.append(xtemp)
		y.append(ytemp)

	# Plot the time series
	plot(x, y, 'b,')
	show()


if __name__ == '__main__':
	main(1.4, 0.35)
	main(1.5, 0.45)

