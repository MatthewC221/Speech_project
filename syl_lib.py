#!/usr/bin/python

"""
Library form of syl.py, doesn't break syllables up but returns RMS and 
voiced/unvoiced ratios. Also has other useful functions
"""

import sys
import numpy as np
import scipy.io.wavfile
from scipy import signal
import wave
import matplotlib.pyplot as plt
import math

def calculate_mean(arr, start, end):
	"""
	Used for calculating the mean ratio of a syllable 
	param @arr: array to calculate (will be the ratio arr)
	param @start: start ind
	param @end: end ind

	return @mean
	"""
	cur = 0
	for i in range(start, end):
		cur += arr[i]

	return (float(cur) / (end - start))

def split_wav(times, file_name):
	"""
	Splits up a .wav file into multiple .wav files based on time
	e.g. times=[1.0, 2.0, 3.0], there will be 4 .wav files (0->1, 1->2, 3->end)
	param @times: the times to split the file into
	param @file_name: the file to split
	"""	

	origAudio = wave.open(file_name,'r')
	frameRate = origAudio.getframerate()
	nChannels = origAudio.getnchannels()
	sampWidth = origAudio.getsampwidth()

	S = 0
	start = float(S)
	times.append(10) 						# For cutting off the last syllable

	for i in range(len(times)):
		end = float(times[i])
		origAudio.setpos(int(start*frameRate))
		chunkData = origAudio.readframes(int((end-start)*frameRate))
		origAudio.tell()

		output = "f" + str(i) + ".wav"
		chunkAudio = wave.open(output,'w')
		chunkAudio.tell()
		chunkAudio.setnchannels(nChannels)
		chunkAudio.setsampwidth(sampWidth)
		chunkAudio.setframerate(frameRate)
		chunkAudio.writeframes(chunkData)
		chunkAudio.close()
		start = end

def periodogram(arr):
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

def unvoiced(x, time_step, window_size):
	"""
	Attempts to separate voiced and unvoiced components
	param @x: PCM signal
	param @time_step: the time between each window 

	return @ratio: ratio of unvoiced to voiced windows

	Steps
	1. Get windowed FFT with hanning window
	2. Calculate power spectrum = P
	3. Log power spectrum 10log10(p) = L
	4. Rolling median of L of size X = M
	5. Calculate ratio of total to voiced (sum p[x] / sum p[x] where l[x] > m[x] + 8)
	"""

	num_win = int(math.ceil(x.size / (2 * time_step)) + 1)	# Number of windows
	size = (window_size / 2) + 1 

	power = np.zeros((num_win, size))
	win_count = 0

	# Get power spectrum of windowed FFT with hanning window
	for i in range(0, x.size / 2, time_step):
		windowed_x = np.zeros(window_size)
		count = 0
		for j in range(i, min(i + window_size, x.size / 2)):
			windowed_x[count] = x[j][0]
			count += 1 
		W = scipy.signal.hanning(window_size, False)
		power[win_count] = periodogram(scipy.fft(windowed_x * W))
		win_count += 1	

	log_power = 10 * np.log10(power)

	median_len = 4 							# Rolling of median 21
	median_arr = np.zeros((num_win, size))

	for i in range(num_win):
		for j in range(median_len, size - median_len):
			cur = []
			for k in range(j - median_len, j + (median_len + 1)):
				cur.append(log_power[i][k])
			cur.sort()
			median_arr[i][j] = cur[median_len]

	ratio = np.zeros(num_win)
	for i in range(num_win):
	 	tot = sum(power[i])
	 	tot_voiced = 0
		for j in range(size):
			if (log_power[i][j] > median_arr[i][j] + 8):
				tot_voiced += power[i][j]

		ratio[i] = float(tot_voiced) / tot

	return ratio

def returnVoicedRMS((fs, x)):
	"""
	Returns RMS, and voiced ratio component 
	param @x: PCM signal
	param @time_step: the time between each window 

	return @ratio: ratio of unvoiced to voiced windows

	Steps
	1. Get windowed FFT with hanning window
	2. Calculate power spectrum = P
	3. Log power spectrum 10log10(p) = L
	4. Rolling median of L of size X = M
	5. Calculate ratio of total to voiced (sum p[x] / sum p[x] where l[x] > m[x] + 8)
	"""

	######################################################
	# STEP 1 prep: calculating step size and window size #
	######################################################

	step_ms = 16 											# 16 m sec steps
	time_step = int((float(step_ms) / 1000) * fs)			# Number of samples stepped 
	window_size = time_step * 4                     		# Number of samples in window

	# Refer to equation f(b, t) = x[b * 705 + t] (b >= 0, 0 <= t < 2820) 
	RMS = np.zeros((x.size / (time_step * 2) + 1, 2))
	count = 0

	#####################################################################
	# STEP 1 & 2: splitting up into blocks and calculating RMS for each #
	#####################################################################

	for i in range(0, x.size / 2, time_step):
		mean_squared_1 = 0
		mean_squared_2 = 0
		upper = min(i + window_size, (x.size / 2) - 1)	
		for j in range(i, upper):
			mean_squared_1 += x[j][0] ** 2
			mean_squared_2 += x[j][1] ** 2
		RMS[count][0] = math.sqrt(mean_squared_1 / window_size)
		RMS[count][1] = math.sqrt(mean_squared_2 / window_size)
		count += 1

	ratio = unvoiced(x, time_step, window_size)

	########################################## 
	# UNUSED STEP: calculating mode + median #
	##########################################
	return (RMS, ratio)

def zero_crossings(fs, x):
	"""
	Returns the zero crossing count of the time domain (consistent with the window)

	param @fs: sampling frequency
	param @x: time domain signal

	return @zero_cross: the amount of times the time domain crosses zero 
	"""

	step_ms = 16 											# 16 m sec steps
	time_step = int((float(step_ms) / 1000) * fs)			# Number of samples stepped 
	window_size = time_step * 4                     		# Number of samples in window

	# Refer to equation f(b, t) = x[b * 705 + t] (b >= 0, 0 <= t < 2820) 
	crossing_counts = np.zeros((x.size / (time_step * 2) + 1))
	count = 0

	for i in range(0, x.size / 2, time_step):
		windowed_x = np.zeros(window_size)
		tmp = 0
		for j in range(i, min(i + window_size, x.size / 2)):
			windowed_x[tmp] = x[j][0]
			tmp += 1 

		zero_cross = np.where(np.diff(np.signbit(windowed_x)))[0]
		crossing_counts[count] = zero_cross.size
		count += 1

	return crossing_counts