#!/usr/bin/python

"""
Library form of mfcc.py, doesn't classify syllables up but returns MFCC coefficients and 
has other useful functions
"""

import sys
import numpy as np
import scipy.io.wavfile
from scipy import signal
import wave
import matplotlib.pyplot as plt
import math

# Currently to classify syllables: ./mfcc.py <.wav> 0 -1 <correct_wav>
# To get average MFCC's: ./mfcc.py <path_to_wav> F <number_wav> F

E = 2.7182818284590452353
rem_size = 12
num_files = -1

class SignalProcessing:
	def compare(self, mfcc, right_syl):
		""" 
		Compares the MFCC values calculated with the reference data ones using
		mean squared error and curve fitting
		param @coeff: calculated MFCC
		"""
		min_E = sys.maxint
		saved = np.zeros(rem_size)
		name = "-1"

		top = []

		f = open("data.txt", "r")
		for line in f:
			parse = line.split(":")
			coeff = parse[1].split(",")

			ref = np.zeros(rem_size)
			count = 0
			for tmp in coeff:
				ref[count] = float(tmp)
				count += 1

			# Taking difference in jumps
			jmp_diff = 0 
			for i in range(1, ref.size):
				jmp1 = mfcc[i] - mfcc[i - 1]
				jmp2 = ref[i] - ref[i - 1]
				jmp_diff += (jmp1 - jmp2)

			E, alpha = self.E_calculation(mfcc, ref)
			if (E < min_E):
				min_E = E
				saved = np.copy(ref)
				name = parse[0]

			if (parse[0] == right_syl):
				plt.plot(ref, label=parse[0], linewidth=5.0)

			top.append((parse[0], E))
		
		top = sorted(top, key=lambda x: x[1])
		print "The top three elements are: "
		for i in range(0, 3):
			print top[i]

		"""
		plt.plot(mfcc, label="Current MFCC", linewidth=5.0)
		plt.plot(saved, label="Closest syllable: " + name)
		plt.legend()
		plt.ylabel("Amplitude")
		plt.xlabel("Coefficients")
		plt.show()
		"""

	def periodogram(self, arr):
		"""
		Calculates the periodogram estimate of power spectrum
		param @arr: np.array of fft window

		return @ret: the periodogram estimate of power spectrum
		"""
		ret_size = (arr.size / 2) + 1
		ret = np.zeros(ret_size)
		mult = (float(1) / ret_size)

		for i in range(ret_size):
			ret[i] = mult * abs(arr[i] ** 2)

		return ret

	def E_calculation(self, ave, ref):
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

		# print ref 
		# print ave

		for i in range(ave.size):
			num += (ave[i] * ref[i])
			denom += (ref[i] ** 2)

		alpha = float(num) / denom

		for i in range(ave.size):
			E += (ave[i] - (ref[i] * alpha)) ** 2

		return (E, alpha)

	def mel_scaled(self, arr):
		"""
		Convers to mel scale from frequency
		param @arr: np.array of frequency

		return @ret: np.array of mel scaled frequency
		"""
		ret = np.zeros(arr.size)
		for i in range(arr.size):
			ret[i] = 2595 * np.log10(1 + (arr[i] / 700))

		return ret

	def mel_freq(self, num):
		"""
		Coverts frequency to mel frequency
		param @num: frequency 

		return @mel_num: mel frequency of num
		"""
		return (1125 * math.log((1 + float(num) / 700), E))

	def inverse_mel_freq(self, num):
		"""
		Returns inverse mel conversion
		param @num: the current mel number

		return @inv_num: inverse mel of num
		"""
		return (700 * (math.exp(num / 1125) - 1))

	def create_filter(self, num_filter, min_freq, max_freq, size, FS):
		"""
		Creates triangular filters, returns filter ranges and size
		param @num_filter: number of triangular filters
		param @min_freq: min_freq we want to capture
		param @max_freq: max_freq we want to capture, normally fs/2
		param @size: fft_size
		param @FS: sample rate of signal

		return @Hm: the triangular filters (np.array of each triangle)
		return @f: the filter ranges (e.g. [[9, 12], [10, 13], [12, 15])
		"""
		min_val = self.mel_freq(min_freq)
		max_val = self.mel_freq(max_freq)
		num_filter = num_filter

		fft_size = size
		fs = FS

		banks = np.zeros(num_filter)
		points = (max_val - min_val) / (num_filter - 1)

		count = 0
		i = min_val 

		while (i <= max_val):
			banks[count] = i
			count += 1
			i += points

		f = np.zeros(banks.size)
		for i in range(banks.size):
			banks[i] = self.inverse_mel_freq(banks[i])
			f[i] = np.floor((fft_size + 1) * banks[i] / fs)

		Hm = np.zeros([num_filter - 2, fft_size // 2 + 1])

		for j in range(0, num_filter - 2):
			# print "range = " + str(f[j]) + ", to " + str(f[j + 1])
			for i in range(int(f[j]), int(f[j+1])):
				Hm[j, i] = (i - f[j]) / (f[j + 1] - f[j])
			for i in range(int(f[j+1]), int(f[j+2])):
				Hm[j, i] = (f[j+2] - i) / (f[j+2] - f[j+1])

		return (Hm, f)

	def compute_mfcc_FULL(self, file_name, FL, window_size, include, mean):
		"""
		Computes the mfcc of the ENTIRE file, adds the current signal to the mean and plots it
		param @file_name: the file we compute mfcc on
		param @FL: the current index we insert into mean (mean[i] = cur_MFCC)
		param @window_size: window size we take the fft on
		param @include: whether the current signal is the data or the reference
		param @mean: np.array to fill up
		"""
		(fs, x) = scipy.io.wavfile.read(file_name)

		""" First step, block process fft using a hanning window """
		step_ms = 25 									# 25 msec steps
		overlap = fs / 100
		window_size = 1024

		num_win = int(math.ceil(x.size / (2 * overlap)) + 1)	# Number of windows
		size = (window_size / 2) + 1 							# Size = 513 (1024 / 2 + 1)

		windowed_fft = np.zeros((num_win, size))

		win_count = 0
		for i in range(0, x.size / 2, overlap):
			windowed_x = np.zeros(window_size)
			count = 0
			for j in range(i, min(i + window_size, x.size / 2)):
				windowed_x[count] = x[j][0]
				count += 1 
			W = scipy.signal.hanning(window_size, False)
			""" Second step, calculate periodogram estimate of power spectrum """
			windowed_fft[win_count] = self.periodogram(scipy.fft(windowed_x * W))
			win_count += 1

		""" Third step, apply mel filterbank to power spectra """
		num_filter = 28
		(Hm, f) = self.create_filter(num_filter, 300, fs / 2, window_size, fs)

		windows = np.zeros((num_filter - 2, 2))			# The filter ranges, e.g. 19->30, 24->36, 30->43

		# Creating the window ranges, e.g. [[9, 12],[12, 15], [15, 18]]
		for i in range(num_filter - 2):
			windows[i] = ((f[i], f[i + 2]))

		bank_energies = np.zeros(num_filter - 2)

		for i in range(num_win):
			# Computing energies, using upper bound and lower bounds
			for j in range(num_filter - 2):
				LB = int(windows[j][0]) + 1
				UB = int(windows[j][1])

				for k in range(LB, UB):
					bank_energies[j] += (Hm[j][k] * windowed_fft[i][k])

		""" Fourth step, take logarithm of all filterbank energies """
		bank_energies = np.log10(bank_energies)

		""" Fifth step, take DCT-2 of filterbank energies """
		dct_energy = scipy.fftpack.dct(bank_energies, 2)
		
		coeff = np.zeros(rem_size)

		for i in range(2, 14):
			coeff[i - 2] = dct_energy[i]

		if (include == True):
			mean[FL] = coeff
			plt.plot(coeff, label=file_name)
		else:		# If false, this is the reference signal, compute MSE
			# plt.plot(coeff, label=file_name, linewidth=4.0)
			return coeff

	def compute_mfcc_BLOCK(self, window, fs=44100):
		"""
		Computes the mfcc of the WINDOW (PCM format), adds the current signal to the mean and plots it
		param @window: the window to compute MFCC's on
		"""

		#########################################################
		# STEP 1: create a hanning window and multiply with PCM #
		#########################################################		

		W = scipy.signal.hanning(window.size, False)
		windowed_fft = self.periodogram(scipy.fft(window * W))

		######################################################
		# STEP 2: Apply mel filterbank to power spectra      #
		######################################################

		num_filter = 28
		(Hm, f) = self.create_filter(num_filter, 300, fs / 2, window_size, fs)

		windows = np.zeros((num_filter - 2, 2))			# The filter ranges, e.g. 19->30, 24->36, 30->43

		# Creating the window ranges, e.g. [[9, 12],[12, 15], [15, 18]]
		for i in range(num_filter - 2):
			windows[i] = ((f[i], f[i + 2]))

		bank_energies = np.zeros(num_filter - 2)

		for i in range(num_win):
			# Computing energies, using upper bound and lower bounds
			for j in range(num_filter - 2):
				LB = int(windows[j][0]) + 1
				UB = int(windows[j][1])

				for k in range(LB, UB):
					bank_energies[j] += (Hm[j][k] * windowed_fft[i][k])

		################################################
		# STEP 3: Take log of filterbank energies      #
		################################################

		bank_energies = np.log10(bank_energies)

		################################################
		# STEP 4: Take DCT of filterbank energies      #
		################################################
		dct_energy = scipy.fftpack.dct(bank_energies, 2)
		
		coeff = np.zeros(rem_size)

		for i in range(2, 14):
			coeff[i - 2] = dct_energy[i]

		return coeff