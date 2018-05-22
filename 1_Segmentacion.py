#Segmentacion

import librosa
import sys		
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

fileName = sys.argv[1]
fileJustName = fileName.split('.')[-2]
if not os.path.exists(fileJustName):
		os.makedirs(fileJustName)

y, sr = librosa.load(fileName)
#onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, backtrack=False)
#print (onset_frames)
onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                        hop_length=512,
                                         )
#print (onset_env)
#onset_bt = librosa.onset.onset_backtrack(onset_raw, onset_env)
#print (onset_raw)
peaks = librosa.util.peak_pick(onset_env, 3, 10, 7, 7, 0.20, 0.01)
#print (peaks)
times = librosa.frames_to_samples(peaks)
#print (times)

file = open(fileJustName + '_durs.txt', 'w')
amountOfSegments = times.size
for cont in range(amountOfSegments - 1):
	posFrameInit = times[cont]
	posFrameEnd = times[cont + 1]
	duracionSeg = posFrameEnd - posFrameInit
	#print("duracion de segmento = " + str(duracionSeg))
	file.write(str(duracionSeg) + "\n")
	librosa.output.write_wav(fileJustName + '/' + fileJustName + '_' 
		+ '{:05d}'.format(cont) + ".wav", y[posFrameInit:posFrameEnd], sr)
file.close()
posFrameInit = times[amountOfSegments-1]
posFrameEnd = y.size
librosa.output.write_wav(fileJustName + '/' + fileJustName + '_' 
	+ '{:05d}'.format(amountOfSegments-1) + ".wav", y[posFrameInit:posFrameEnd], sr)

times = librosa.samples_to_time(times)
#print times

amountOfSegments = times.size

file = open(fileJustName + '_times.txt', 'w')
for cont in range(amountOfSegments -1):
	posFrameInit = times[cont]
	file.write(str(posFrameInit) + "\n")
file.close()

print ("numero de segmentos = " + str(amountOfSegments))

#Plot
timess = librosa.frames_to_time(np.arange(len(onset_env)),
                               sr=sr, hop_length=512)
plt.figure()
ax = plt.subplot(2, 1, 2)
D = librosa.stft(y)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         y_axis='log', x_axis='time')
plt.subplot(2, 1, 1, sharex=ax)
plt.plot(timess, onset_env, alpha=0.8, label='Onset strength')
plt.vlines(timess[peaks], 0,
           onset_env.max(), color='r', alpha=0.8,
           label='Selected peaks')
plt.legend(frameon=True, framealpha=0.8)
plt.axis('tight')
plt.tight_layout()
plt.show()

