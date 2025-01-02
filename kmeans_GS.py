#####################################################
#   K-Means Clustering Example Code                 #
#   Author: Grace Stapkowski                        #
#   Date Created: 7/29/24                           #
#####################################################

# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# make test data
# Note: assigning an int to random_states makes the blobs reproducible
features, true_labels = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=2.75,
    random_state=42
    )

# plot raw test data
fig = plt.figure(0)
plt.grid(True)
plt.scatter(features[:,0],features[:,1])
plt.show()

# create model 
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=20,
    max_iter=300,
    random_state=None
    )

# run KMeans
kmeans.fit(features)
y_kmeans = kmeans.predict(features)

# plot resutling centroids
fig = plt.figure(0)
plt.grid(True)
plt.scatter(features[:,0],features[:,1], c=y_kmeans, s=50)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red',
            marker="X")
plt.show()

# print report on clustering 
print("Number of iterations to convergence: " + str(kmeans.n_iter_))
