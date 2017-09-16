#!/usr/bin/python

import sys
import numpy as np
import scipy.io.wavfile
import wave
import matplotlib.pyplot as plt

if (len(sys.argv) == 2):
	(fs, x) = scipy.io.wavfile.read(sys.argv[1])

	"""
	fft_x = scipy.fft(x)
	period = abs(fft_x ** 2) / fft_x.size

	plt.plot(period)
	plt.ylabel("Power spectrum (dB)")
	plt.xlabel("Frequency")

	plt.show()

	"""
	
	threshold = 12

	# Yes should have higher density for higher frequencies due to the s
	interval1 = (x.size * 5000 / fs)
	interval2 = (x.size * 11025 / fs)

	fft_x = abs(scipy.fft(x))
	f = np.sum(fft_x[1:interval1]) / np.sum(fft_x[interval1:interval2])

	if (f < threshold):
		print "result = yes"
	else:
		print "result = no"