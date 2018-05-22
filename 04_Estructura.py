#Estructura

import sys
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

FACTOR_DE_REDUCCION = 256
audioname = sys.argv[1]
clases = open(audioname + "_clases.txt")
durs = open(audioname + "_durs.txt")
clasescontenttmp = clases.readlines()
durscontenttmp = durs.readlines()
clasescontent = []
durscontent = []
for item in clasescontenttmp:
	clasescontent.append(int(item.split(" ")[0]))
for item in durscontenttmp:
	durscontent.append(int(item) / FACTOR_DE_REDUCCION)

print(clasescontent)
print(durscontent)

cantClases = max(clasescontent) + 1
reparticionDeClases = np.zeros(cantClases)
nombreDeClases = []

estructura = []
for clase, dur in zip(clasescontent, durscontent):
	reparticionDeClases[clase] += dur
	estructura.append([clase] * dur)
estructura = [item for sublist in estructura for item in sublist]

for n in range(cantClases):
	nombreDeClases.append("Clase " + str(n))

reparticionDeClases = reparticionDeClases / sum(reparticionDeClases)

eventData = [[] for i in range(cantClases)]
posEnSamplesCFR = 0
for clase, dur in zip(clasescontent, durscontent):
	for i in range(dur):
		eventData[clase].append(posEnSamplesCFR + i)
	posEnSamplesCFR += dur

print(posEnSamplesCFR)

cantClasessec = [] 
for i in range(cantClases):
	cantClasessec.append((cantClases - cantClases) + i)

#PLot
plt.figure(1)
plt.subplot(121)             # the first subplot in the first figure
plt.xlabel('Segundos')
plt.ylabel('Clases')
samplesReales = posEnSamplesCFR * FACTOR_DE_REDUCCION
segundosINT = (int)(samplesReales / 22050.0)
puntos = range(segundosINT)[0::10]
posiciones = np.array(puntos) * 22050 / FACTOR_DE_REDUCCION
plt.xticks(posiciones,puntos)
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.yticks(cantClasessec, cantClasessec)
plt.eventplot(eventData, linelengths=1, lineoffsets=1)
limon = plt.subplot(122)             # the second subplot in the first figure
plt.pie(reparticionDeClases, labels=nombreDeClases, autopct='%1.1f%%')
plt.axis('equal')
plt.show()
