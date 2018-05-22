#Clasificacion

import tensorflow as tf
import numpy as np
import sys
import glob

k = 8
max_iterations = 100
folder = sys.argv[1]

def loadData(xs, names):
    for fileName in glob.glob(folder + '/*.txt'):
        file = open(fileName, 'r')
        print(fileName)
        names.append(fileName)
        content = file.readlines()
        data = content[0].split(',')
        data = map(float, data)
        print(data)
        xs.append(data)

def get_dataset():
    xs = list()
    names = list()
    loadData(xs, names)
    xs = np.asmatrix(xs)
    return xs, names

def initial_cluster_centroids(X, k):
    return X[0:k, :]

def assign_cluster(X, centroids):
    expanded_vectors = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)
    return mins

def recompute_centroids(X, Y):
    sums = tf.unsorted_segment_sum(X, Y, k)
    counts = tf.unsorted_segment_sum(tf.ones_like(X), Y, k)
    return sums / counts

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    X, names = get_dataset()
    centroids = initial_cluster_centroids(X, k)
    i, converged = 0, False
    while not converged and i < max_iterations:
        i += 1
        Y = assign_cluster(X, centroids)
        centroids = sess.run(recompute_centroids(X, Y))
    results = zip(sess.run(Y), names)
    results.sort(key=lambda tup: tup[1])
    file = open(folder + "_clases.txt", "w")
    for res in results:
        print("clase " + str(res[0]) + " segmento " + res[1])
        segmento = res[1].split("_")[1].split(".")[0]
        file.write(str(res[0]) + " " + segmento + "\n")
    file.close()


