# plot periodic orbits and iterations for the logistic map
#
import numpy as np
import matplotlib.pyplot as plt

# logistic map is f(x) = mu*x*(1-x)  with mu in (0,4)
def logistic(x,r):
	return r * x * (1.0-x)


# fill an array with iteration n1 to n2 of the logistic map starting with x0
# and with parameter mu
def fillit(n1,n2,x0,mu):
	x = x0  # initial x value
	z = np.linspace(0.0,1.0,n2-n1)  # create an array
	for i in range(0,n1):   # do n1 iterations
		x = logistic(x,mu)
	for i in range(0,n2-n1):   # fill n2-n1 iterations
		x = logistic(x,mu)
		z[i] = x
	return z  # returning the array


# plot the iterated logistic map for nmu number of mu values
def mkplot(mu_min,nmu):  # nmu is number of mu values to use, mu_min range
	mu_max = 4.0  # maximum mu value
	muarr = np.linspace(mu_min,mu_max,nmu)
	n1=100  # specify iteration range
	n2=200
	x0=0.5  # initial x
	for i in range(0,nmu):
		mu = muarr[i]
		y=fillit(n1,n2,x0,mu)  # get the array of iterations
		x=y*0.0 + mu   # dummy x value is all mu
		plt.plot(x,y,'ko',markersize=1) # k=black, plot small points
	plt.show()


if __name__ == '__main__':
	plt.figure()
	plt.xlabel(r'r', fontsize=20)
	plt.ylabel('x', fontsize=20)
	mu_min = 2.9
	plt.axis([mu_min, 4.0, 0, 1.0])
	# this makes the plot!
	mkplot(mu_min, 1000)
