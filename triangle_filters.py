#!/usr/bin/python

import sys
import numpy as np
import scipy.io.wavfile
from scipy import signal
import wave
import matplotlib.pyplot as plt
import math

# Creating triangle filters, specifically with: fs=16000, fft_size=512, 28 filters
# Creates a nice plot to help visualise

E = 2.7182818284590452353

def mel_freq(num):
	return (1125 * math.log((1 + float(num) / 700), E))

def inverse_mel_freq(num):
	return (700 * (math.exp(num / 1125) - 1))

def create_filter(num_filter, min_freq, max_freq):
	min_val = mel_freq(min_freq)
	max_val = mel_freq(max_freq)
	num_filter = num_filter

	fft_size = 512
	fs = 44100

	banks = np.zeros(num_filter)
	points = (max_val - min_val) / (num_filter - 1)
	count = 0
	i = min_val 

	# Creating the banks of mel frequencies
	while (i <= max_val):
		banks[count] = i
		count += 1
		i += points

	f = np.zeros(banks.size)
	for i in range(banks.size):
		banks[i] = inverse_mel_freq(banks[i])
		f[i] = np.floor((fft_size + 1) * banks[i] / fs)

	print f
	Hm = np.zeros([num_filter - 2, fft_size // 2 + 1])

	for j in range(0, num_filter - 2):
		for i in range(int(f[j]), int(f[j+1])):
			Hm[j, i] = (i - f[j]) / (f[j + 1] - f[j])
		for i in range(int(f[j+1]), int(f[j+2])):
			Hm[j, i] = (f[j+2] - i) / (f[j+2] - f[j+1])


	for i in range(num_filter - 2):
		plt.plot(Hm[i])

	plt.ylabel("Amplitude")
	plt.xlabel("FFT sample")
	plt.show()

create_filter(28, 300, 22050)