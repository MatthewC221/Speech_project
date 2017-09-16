#!/usr/bin/python

import sys
import numpy as np
import scipy.io.wavfile
from scipy import signal
import wave
import matplotlib.pyplot as plt
import math

E = 2.7182818284590452353
rem_size = 12

class SignalProcessing:
	def periodogram(self, arr):
		"""
		Calculates the periodogram estimate of power spectrum
		param @arr: np.array of fft window
		"""
		ret_size = (arr.size / 2) + 1
		ret = np.zeros(ret_size)
		mult = (float(1) / ret_size)

		for i in range(ret_size):
			ret[i] = mult * abs(arr[i] ** 2)

		return ret

	def mel_scaled(self, arr):
		"""
		Convers to mel scale from frequency
		param @arr: np.array of frequency
		"""
		ret = np.zeros(arr.size)
		for i in range(arr.size):
			ret[i] = 2595 * np.log10(1 + (arr[i] / 700))

		return ret

	def mel_freq(self, num):
		"""
		Coverts frequency to mel frequency
		param @num: frequency 
		"""
		return (1125 * math.log((1 + float(num) / 700), E))

	def inverse_mel_freq(self, num):
		"""
		Returns inverse mel conversion
		param @num: the current mel number
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

	def compute_mfcc(self, file_name, FL, window_size, include, mean):

		"""
		Computes the mfcc, adds the current signal to the mean and plots it
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
		else:
			plt.plot(coeff, label=file_name, linewidth=4.0)

def main():
	if (len(sys.argv) != 4):
		print "Usage ./mfcc.py <.wav file> <compared_wav> <number_wav>"
	else:
		num_files = int(sys.argv[3]) + 1
		signal = SignalProcessing()
		mean = np.zeros((num_files, 12))

		for FL in range(num_files):						# Run for multiple files
			file_name = sys.argv[1] + str(FL) + ".wav"
			signal.compute_mfcc(file_name, FL, 1024, True, mean)

		if (sys.argv[2] != "F"):
			compared = sys.argv[2] + ".wav"
			signal.compute_mfcc(compared, 0, 1024, False, mean)

		total_mean = np.zeros(rem_size)
		for i in range(num_files):
			for j in range(rem_size):
				total_mean[j] += (mean[i][j] / (num_files))

		plt.plot(total_mean, label="Mean of files", linewidth=5.0)
		plt.legend()
		plt.ylabel("Amplitude")
		plt.xlabel("Coefficients")
		plt.show()

if __name__ == "__main__":
	main()