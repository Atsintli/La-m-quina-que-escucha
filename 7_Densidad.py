# Analisis de densidad

import librosa
import sys
import os
import numpy as np
import glob
import math

fileName1 = sys.argv[1]
fileName2 = sys.argv[2]

#slice time are
FramesSec = 22050.0
SecSlice = 10 
SecSliceFrames = SecSlice * FramesSec

def getBlocks(filename):
	y, sr = librosa.load(filename)
	dur = y.size / FramesSec
	blocks_amount = (int)(math.ceil(dur / SecSlice))
	print(dur)
	print(blocks_amount)
	onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='samples') / FramesSec
	blocks = np.zeros(blocks_amount)
	for onset in onset_frames:
		indexofonset = (int)(onset / SecSlice)
		blocks[indexofonset] = blocks[indexofonset] + 1
	return blocks

b1 = getBlocks(fileName1)
b2 = getBlocks(fileName2)

#iguala tamanios
difsize = b1.size - b2.size
if difsize < 0:
	b1 = np.append(b1, np.zeros(abs(difsize)))
if difsize > 0:
	b2 = np.append(b2, np.zeros(difsize))

dif = np.fabs(b1 - b2) #Compute the absolute values element-wise.
dif = np.sum(dif)
print ("La diferencia entre las densidades sonoras de los archivos es de " + str(dif))




