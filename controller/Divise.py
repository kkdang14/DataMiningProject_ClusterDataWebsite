import matplotlib.colors as mcolors
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt


def DiviseClustering(data, current_cluster, target_clusters, depth=0, tree=None):
    if tree is None:
        tree = []

    if len(np.unique(current_cluster)) >= target_clusters:
        return current_cluster, tree

    for cluster_id in np.unique(current_cluster):
        cluster_data = data[current_cluster == cluster_id]

        if len(cluster_data) > 1:
            # Compute dissimilarity (pairwise distance) matrix
            dissimilarity_matrix = pairwise_distances(cluster_data)

            # Identify the most dissimilar point (with highest avg dissimilarity)
            avg_dissimilarity = dissimilarity_matrix.mean(axis=1)
            most_dissimilar_point = np.argmax(avg_dissimilarity)

            # Split the cluster based on dissimilarity
            new_labels = np.zeros(len(cluster_data))
            new_labels[most_dissimilar_point] = 1  # Assign the most dissimilar point to a new cluster

            # Measure distance from the most dissimilar point
            distances = dissimilarity_matrix[most_dissimilar_point]
            new_labels[distances >= np.median(distances)] = 1  # Cluster the other points

            # Relabel the clusters
            new_cluster_ids = new_labels + np.max(current_cluster) + 1
            current_cluster = current_cluster.copy()
            current_cluster[current_cluster == cluster_id] = new_cluster_ids

            # Track the split in the tree structure
            tree.append((depth, cluster_id, np.max(current_cluster) - 1, np.max(current_cluster)))

            # Stop if target clusters are achieved
            if len(np.unique(current_cluster)) >= target_clusters:
                break

            # Recursively split the resulting clusters
            current_cluster, tree = DiviseClustering(data, current_cluster, target_clusters, depth + 1, tree)

    return current_cluster, tree



