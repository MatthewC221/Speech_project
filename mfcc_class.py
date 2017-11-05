#!/usr/bin/python

# MFCC classifier for one file

from mfcc_lib import *
from syl_lib import *
from custom_mfcc import *

import sys
import numpy as np
import scipy.io.wavfile
from scipy import signal
import wave
import matplotlib.pyplot as plt
import math


# Consider how the MFCC is changing over time!

if (len(sys.argv) != 2):
	print "Usage ./experimental.py <.wav file>"
else:
	FN = sys.argv[1]
	(fs, x) = scipy.io.wavfile.read(FN)
	coeff = mfcc(x)
	for i in xrange(coeff.size / coeff[0].size):
		compare(coeff[i], coeff[0].size)