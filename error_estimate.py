#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt

def E_calculation(ave, ref):

	"""
	Attempts to fit the ref signal to the average signal (using MSE)
	param @ave: the average signal 
	param @ref: the reference signal

	return @E: 
	return @Alpha:
	"""

	# Alpha = (sum c[n] * m[n]) / (sum c[n] ** 2)
	num = 0
	denom = 0
	E = 0

	for i in range(ave.size):
		num += (ave[i] * ref[i])
		denom += (ave[i] ** 2)

	print "Num = " + str(num) + ", denom = " + str(denom)
	alpha = float(num) / denom

	for i in range(ave.size):
		E += (ave[i] - (ref[i] * alpha)) ** 2

	return (E, alpha)

def main():

	ave = np.array([1,2,3,4,-5])
	ref = np.array([10,20,30,40,50])

	E, Alpha = E_calculation(ave, ref)
	MSE = np.mean((ave - ref) ** 2)

	print "Error = " + str(E) + ", alpha = " + str(Alpha)
	print "Mean squared error = " + str(MSE)
	plt.plot(ave, label="Average")
	plt.plot(ref, label="Reference")
	plt.legend()
	plt.ylabel("Amplitude")
	plt.xlabel("Samples")
	plt.show()

if __name__ == "__main__":
	main()