#Extraccion

import librosa
import glob
import sys
import numpy as np

def extract(fileName):
	fileJustName = fileName.split('.')[-2]
	print(fileJustName + ' Proceced')
	y, sr = librosa.load(fileName)
	stft = np.abs(librosa.stft(y))
	mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2).T, axis=0)
	#chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0) 
	#mel = np.mean(librosa.feature.melspectrogram(y, sr=sr).T,axis=0)
	#contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
	#bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)
	#cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
	#flatness = np.mean(librosa.feature.spectral_flatness(y=y).T, axis=0)
	#allFeature = np.concatenate((flatness, mfccs, contrast, chroma, bandwidth, cent))
	allFeature = map(str, mfccs)
	file = open(fileJustName + '.txt', 'w')
	file.write(', '.join(allFeature))
	file.close()
folder = sys.argv[1]
for file in glob.glob(folder + '/*.wav'):
	extract(file)


