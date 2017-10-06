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

if (len(sys.argv) != 2):
	print "Usage ./experimental.py <.wav file>"
else:
	FN = sys.argv[1]
	(fs, x) = scipy.io.wavfile.read(FN)

	RMS, ratio = returnVoicedRMS((fs, x))
	plt.subplot(2, 1, 1)
	plt.plot(RMS)
	plt.ylabel("RMS")
	plt.xlabel("Window")

	plt.subplot(2, 1, 2)
	plt.plot(ratio)
	plt.ylabel("Ratio V:UV+V")
	plt.xlabel("Window")

	plt.show()