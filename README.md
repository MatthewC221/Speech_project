#### Subsolution to speech recognition MAJOR PROJECT 
*Currently being developed*

#### Two main components are: syllable separation and MFCC computation
##### Syllable separation: 
	Applied RMS block processing on PCM (syl.py)
	./syl.py <wav_name> <plot (0/1)>
	If you had test.wav and you wanted to plot it, you would run it like so
	./syl.py test.wav 1
##### MFCC computation: 
	Applied hanning window and FFT on PCM, compute power of triangular filters
	./mfcc.py <wav_name> <compared_wav> <number_of_wav>
	If you had gen0.wav, gen1.wav, gen2.wav, gen3.wav and then a test.wav
	you would run it like ./mfcc.py gen test 3
##### Other random components:
	yes_no.py (way to separate yes and no)
	triangle_filters.py (created the triangular filters, function inc. in mfcc.py)
	split_wav.py (splits a wav by time, function inc. in mfcc.py)	
##### Data:
The data I have created won't be uploaded onto github, currently has ~50 .wav files
*Example output of ./syl.py little.wav 1*
![Alt text](/little.png?raw=true "Optional Title")

