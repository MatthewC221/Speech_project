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

# Consider how the MFCC is changing over time!

if (len(sys.argv) != 2):
	print "Usage ./experimental.py <.wav file>"
else:
	FN = sys.argv[1]
	(fs, x) = scipy.io.wavfile.read(FN)

	RMS, ratio = returnVoicedRMS((fs, x))

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

	# for i in xrange(count):
	#	signal.compare(all_coeff[i])
	# fig1 = plt.figure()
	# ax1 = fig1.add_subplot(111)

	zero_cross = zero_crossings(fs, x)

	f, axarr = plt.subplots(4, sharex=True)
	axarr[0].plot(RMS)
	axarr[0].set_title("RMS")
	axarr[1].plot(ratio)
	axarr[1].set_title("Voiced to unvoiced ratio")
	axarr[2].plot(zero_cross)
	axarr[2].set_title("Zero crossing counts")

	size = all_coeff.size / num_coeff
	for i in xrange(num_coeff):
		cur = np.zeros(size)
		for j in xrange(size):
			cur[j] = all_coeff[j][i]	
		axarr[3].plot(cur, label="MFCC coeff " + str(i))

	axarr[3].set_title("MFCC")
	# axarr[3].legend(loc='upper left')

	plt.xlabel("Window")
	plt.show()