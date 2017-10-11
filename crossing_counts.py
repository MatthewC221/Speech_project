#!/usr/bin/python

import sys
import numpy as np 

def zero_crossings(x):
	"""
	Returns the zero crossing count of the time domain (consistent with the window)

	param @fs: sampling frequency
	param @x: time domain signal

	return @zero_cross: the amount of times the time domain crosses zero 
	"""
	zero_cross = np.where(np.diff(np.signbit(x)))[0]

	return zero_cross.size

print zero_crossings([-1, 1, -1, 1, -1, 1, 1, 0, 0, 1, 1, 1, -1])