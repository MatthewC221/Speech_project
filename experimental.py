#!/usr/bin/python

# A combination of .mfcc and .syl (experimental release)

from mfcc_lib import *
from syl_lib import *

import sys
import numpy as np
import scipy.io.wavfile
from scipy import signal
import wave
import matplotlib.pyplot as plt
import math

num_coeff = 12

if (len(sys.argv) != 2):
	print "Usage ./experimental.py <.wav file>"
else:
	FN = sys.argv[1]
	(fs, x) = scipy.io.wavfile.read(FN)

	RMS, ratio = returnVoicedRMS((fs, x))
	plt.subplot(2, 1, 1)
	plt.plot(RMS)
	plt.ylabel("RMS")

	plt.subplot(2, 1, 2)
	plt.plot(ratio)
	plt.ylabel("Ratio V:UV+V")
	plt.xlabel("Window")

	step_ms = 16 											# 16 m sec steps
	time_step = int((float(step_ms) / 1000) * fs)			# Number of samples stepped 
	window_size = time_step * 4                     		# Number of samples in window

	signal = SignalProcessing()
	num_win = int(math.ceil(x.size / (2 * time_step)) + 1)	# Number of windows
	(filters, Hm) = signal.mfcc_SETUP(window_size)

	all_coeff = np.zeros((num_win, num_coeff))

	# Find the MFCC coefficients at each window (using the same steps and window sizes)
	count = 0
	for i in xrange(0, x.size / 2, time_step):
		window = np.zeros(window_size)
		win_count = 0
		if ((i + window_size) > (x.size / 2)): break
		for j in range(i, i + window_size):
			window[win_count] = x[j][0]
			win_count += 1
		all_coeff[count] = signal.compute_mfcc_BLOCK(window, filters, Hm)
		count += 1

	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	ax1.plot(all_coeff)
	plt.show()