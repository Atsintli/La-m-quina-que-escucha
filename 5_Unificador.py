#Unificador

import librosa
import sys
import numpy as np

NUM_MAX_CLASES = 8
audioname = sys.argv[1]
clases = open(audioname + "_clases.txt")
clasescontenttmp = clases.readlines()
clasescontent = []
for item in clasescontenttmp:
	clasescontent.append(int(item.split(" ")[0]))
print(clasescontent)
for clase in range(NUM_MAX_CLASES):
	print("iterando sobre " + str(clase))
	ele = np.where(np.array(clasescontent)==clase)[0]
	print("indices de clase " + str(clase) + " son ")
	print(ele)
	audiototal = np.array([])
	for elements in ele:
		conStr = '{:05d}'.format(elements)
		nomArchivo = audioname + "/" + audioname + "_" + conStr + ".wav"
		print("leyendo " + nomArchivo)
		y, sr = librosa.load(nomArchivo)
		audiototal = np.append(audiototal,y)
	librosa.output.write_wav(audioname + "/" + audioname + "_CLASE_" 
		+ str(clase) + ".wav", audiototal,sr)
