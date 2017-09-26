#!/usr/bin/python

# Script to separate syllables before classifying them, inspired by Pratt script: 
# https://link.springer.com/content/pdf/10.3758%2FBRM.41.2.385.pdf
# However there are definitely my own elements in there

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
		start = end - 0.05

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

if (len(sys.argv) != 3):
	print "Usage ./syl.py <.wav file> <plot (0/1)>"
else:
	FN = sys.argv[1]
	(fs, x) = scipy.io.wavfile.read(FN)

	"""
	Steps to achieve (based on Pratt script)

	1. First get intensity: energy (root mean square):
	- Choose window, take each sample and square
	- Sum these squares and divide to get mean
	- Take square root of that val 
	volume = 10 * log(C * sum)

	Get intensity over time window of 64msec and 16msec time steps

	2. Peaks are above a threshold, get median intensity and set thres to 2-4 dB above

	3. Inspect the dip before, and consider only peaks with dips of 2-4 dB with respect to current peak

	4. Extract pitch contour, window size of 100msec and 20-msec steps

	5. Remaining peaks are syllables

	"""
	# Step 1
	# No need for FFT, but use RMS to classify window energy. (L2-norm gives too high values such as 2 * 10^8)

	step_ms = 16 									# 16 m sec steps

	time_step = int((float(step_ms) / 1000) * fs)	# Number of samples stepped 
	window_size = time_step * 4                     # Number of samples in window

	# Refer to equation f(b, t) = x[b * 705 + t] (b >= 0, 0 <= t < 2820) 
	RMS = np.zeros((x.size / (time_step * 2) + 1, 2))
	count = 0

	# Get the RMS of the window
	for i in range(0, x.size / 2, time_step):
		mean_squared_1 = 0
		mean_squared_2 = 0
		upper = min(i + window_size, (x.size / 2) - 1)		# For last window
		for j in range(i, upper):
			mean_squared_1 += x[j][0] ** 2
			mean_squared_2 += x[j][1] ** 2
		RMS[count][0] = math.sqrt(mean_squared_1 / window_size)
		RMS[count][1] = math.sqrt(mean_squared_2 / window_size)
		count += 1

	ratio = unvoiced(x, time_step, window_size)
	# End of step 1

	# Step 2: calculate threshold, testing mode / mean / median intensity

	print "----- Testing thresholds -----"

	# Testing mode intensity
	bins = np.arange(0, np.max(RMS), np.max(RMS) / 20)			# Volume level, have 20 bins of vol
	density_1 = np.zeros(bins.size)
	density_2 = np.zeros(bins.size)

	for i in range(RMS.size / 2):
		if (RMS[i][0] > 1.5):
			for j in range(bins.size):
				if (bins[j] > RMS[i][0]):
					density_1[j] += 1
					break
		if (RMS[i][1] > 1.5):
			for j in range(bins.size):
				if (bins[j] > RMS[i][1]):
					density_2[j] += 1
					break
	index_1 = 0
	index_2 = 0
	max_density_1 = 0
	max_density_2 = 0

	for i in range(density_1.size):
		if (density_1[i] > max_density_1):
			max_density_1 = density_1[i]
			index_1 = i
		if (density_2[i] > max_density_2):
			max_density_2 = density_2[j]
			index_2 = i		

	print "Mode density 1 = " + str(bins[index_1])
	print "Mode density 2 = " + str(bins[index_2])

	# Testing median
	chan_1 = np.zeros(RMS.size / 2)
	for i in range(RMS.size / 2):
		chan_1[i] = RMS[i][0]

	median_1 = (np.max(chan_1) / 2) 

	print "Median density = " + str(median_1)

	# Testing average, (only take values that are significant, the silence will lower the mean GREATLY)
	# There is an issue with outliers...


	# Extra step, getting rid of outliers, put everything into bins of 200. We'll establish a good max
	bins = np.arange(0, np.max(RMS) + 200, 200)
	freq = np.zeros(bins.size)

	for i in range(RMS.size / 2):
		j = 0
		while (RMS[i][0] > bins[j]):
			j += 1
		freq[j] += 1

	# Setting upper bound

	max_val = np.max(RMS)
	mean = 0
	count = 0

	for i in range(RMS.size / 2):
		if (RMS[i][0] > max_val / 5):
			mean += RMS[i][0]
			count += 1

	if (mean):
		mean = mean / count 
	# End of step 2

	# Step 3

	# upper_threshold = mean / count
	upper_threshold = np.max(RMS) / 5
	step = mean / 6
	lower_threshold = upper_threshold / 3 		 		

	dip = True									# First syllable doesn't require a dip
	count_ABOVE = 0								# Peaks and dips are maintained for at least ~3 windows 
	count_BELOW = 0								# = 0.016 * 3 = 0.048 seconds
	maintain = 2

	print "----- Final thresholds -----"
	print "The upper threshold = " + str(upper_threshold)
	print "The starting lower threshold = " + str(lower_threshold)

	split = []			# The times for the syllable split

	syl = 0
	i = 0
	outline = np.zeros(RMS.size / 2) 
	dips = np.zeros(RMS.size / 2) 

	"""
	Getting rid of multi peak syllables based on speaking rate
	110-150 WPM, 1.66 syllables per word = 150 * 1.66 = 250 syl p/min = 4.16 syl p/sec
	A syllable every 0.24 seconds at most, round down to 0.08 seconds
	Anything faster is too much, window step = 0.016, therefore has to be > 5 samples difference
	"""

	last_peak = []
	ratio_syl = []
	# The core processing part, determining peaks and dips

	while (len(RMS) > i):
		if (RMS[i][0] > upper_threshold):
			count_ABOVE += 1
			count_BELOW = 0
		elif (RMS[i][0] < lower_threshold):
			count_BELOW += 1
			count_ABOVE = 0
		else:						# If in middle, no peak / dip
			count_BELOW = 0
			count_ABOVE = 0
		# If we're in a possible peak
		# print "RMS[i][0] = " + str(RMS[i][0]) + ", i = " + str(i) + ", count_ABOVE = " + str(count_ABOVE)
		if (dip == True and RMS[i][0] > upper_threshold and count_ABOVE >= maintain):
			# print "Index = " + str(i)
			if (len(last_peak) == 0 or i > last_peak[-1] + 5):
				peak_val = 0
				# Requires a time difference of 5 windows and also a peak in the next
				# 30 windows (A < P > B)
				for j in range(i, min(i + 30, len(RMS))):
					if ((RMS[j][0] > RMS[j - 1][0] and RMS[j][0] > RMS[j + 1][0])):
						break

				peak_val = RMS[j][0]
				lower_threshold = max(upper_threshold / 3, RMS[j][0] - step)
				after = j 
				flag_after = False
				flag_before = False
				dip = False
				before = max(j - 7, 0)	

				# Check if there's a dip in the next 5 windows
				while (after < min(len(RMS), j + 7) or peak_val > RMS[after][0]):
					if (RMS[after][0] < lower_threshold):
						flag_after = True
						break
					after += 1

				# Check if there's a dip in the previous 5 windows
				while (before < j or peak_val > RMS[before][0]):
					if (RMS[before][0] < lower_threshold):
						flag_before = True
						break
					before += 1

				# IFF there is a dip before and after we can proceed
				insert = False
				if (flag_before == True and flag_after == True):
					# print "1. I = " + str(i)
					if (RMS[j][0] > upper_threshold):
						i = j + 3
						count_ABOVE = 0

						calc = calculate_mean(ratio, max(0, j - 3), min(ratio.size, j + 3))
						print "Attempted = " + str(calc) + ", on j = " + str(j)
						if (calc > 0.20):
							ratio_syl.append(calc)
							last_peak.append(j)
							syl += 1
							outline[j] = RMS[j][0]
							insert = True

					# print "2. I = " + str(i)
					# Move over the peak into a dip area
					while (i < len(RMS) - 1):
						if (RMS[i][0] < lower_threshold):
							while (RMS[i][0] > RMS[i + 1][0] and (i < len(RMS) - 2)):
								i += 1
							break
						#if (RMS[i][0] > peak_val):
						#	break
						i += 1
					if (insert == True):
						split.append(i + 3)
				else:
					count_ABOVE = 0
					count_BELOW = 0
				dip = True

		elif (dip == False and count_BELOW >= maintain):
			dip = True	
		i += 1

	print "Syllable count = " + str(syl)

	start = 0

	"""
	for i in range(len(split)):
		mean_ratio = calculate_mean(ratio, start, split[i])
		print "Ratio for syllable " + str(i) + ", = " + str(mean_ratio)
		start = split[i]
	"""

	time_split = [(0.016 * x) for x in split]
	if (len(time_split) > 0):
		del time_split[-1]

	print "Split at times: " + str(time_split)
	print ratio_syl

	# Last step, extract pitch contour using 100msec windows and 20msec time steps
	"""

	step_ms = 20 									# 16 m sec steps
	time_step = int((float(step_ms) / 1000) * fs)	# Number of samples stepped 
	window_size = time_step * 5                     # Number of samples in window

	# Using matplotlibs spectogram
	Sxx = np.zeros((x.size / (time_step * 2) + 1))
	t = np.zeros(Sxx.size)
	f = np.zeros(Sxx.size)

	x_1 = np.zeros(x.size / 2)
	for i in range(x.size / 2):
		x_1[i] = x[i][0]

	count = 0
	for i in range(0, x.size / 2, time_step):
		cur_window = x_1[i:i+window_size]
		f_tmp, t_tmp, Sxx_tmp = signal.spectrogram(cur_window, fs)
		Sxx[count] = np.sum(Sxx_tmp ** 2) / window_size
		count += 1 

	plt.plot(Sxx)

	plt.plot(RMS)
	plt.ylabel('Energy')
	plt.xlabel('Time [sec]')
	plt.show()

	"""
	if (sys.argv[2] == "1"):
		split_wav(time_split, FN)
		plt.plot(chan_1)
		plt.plot(outline)
		plt.ylabel("RMS")
		plt.xlabel("Window")
		plt.show()
