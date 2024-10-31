import matplotlib.colors as mcolors
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt


def DivisiveClustering(data, current_cluster, target_clusters, depth=0, tree=None):
    if tree is None:
        tree = []

    if len(np.unique(current_cluster)) >= target_clusters:
        return current_cluster, tree

    for cluster_id in np.unique(current_cluster):
        cluster_data = data[current_cluster == cluster_id]

        if len(cluster_data) > 1:
            # Compute dissimilarity (pairwise distance) matrix
            dissimilarity_matrix = pairwise_distances(cluster_data, metric='euclidean')

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
            current_cluster, tree = DivisiveClustering(data, current_cluster, target_clusters, depth + 1, tree)

    return current_cluster, tree



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import pdist, squareform
# from sklearn.preprocessing import StandardScaler
# import scipy.cluster.hierarchy as sch

# class DivisiveClustering:
#     def __init__(self, metric='euclidean', standardize=False, stop_at_k=False, trace_level=0):
#         self.metric = metric
#         self.standardize = standardize
#         self.stop_at_k = stop_at_k
#         self.trace_level = trace_level
#         self.clustering_result = None
#
#     def fit(self, x):
#         # Ensure input is a NumPy array
#         if isinstance(x, np.ndarray):
#             self.data = x
#         else:
#             raise ValueError("Input must be a numpy array")
#
#         n = self.data.shape[0]
#
#         # Standardize the data if required
#         if self.standardize:
#             scaler = StandardScaler()
#             self.data = scaler.fit_transform(self.data)
#
#         # Compute dissimilarity matrix
#         diss_matrix = pdist(self.data, metric=self.metric)
#         diss_matrix = squareform(diss_matrix)
#
#         clusters = [list(range(n))]  # Initial cluster includes all samples
#         merge_list = []  # To store merge information
#
#         while len(clusters) < n:
#             # Calculate cluster diameters
#             c_diameters = [np.max(diss_matrix[cluster][:, cluster]) for cluster in clusters]
#             max_cluster_idx = np.argmax(c_diameters)  # Index of the cluster with maximum diameter
#
#             # Compute average dissimilarity within the cluster
#             cluster_data_indices = clusters[max_cluster_idx]
#             cluster_data = self.data[cluster_data_indices]
#             cluster_diss = pdist(cluster_data, metric=self.metric)
#             avg_dissimilarity = np.mean(cluster_diss)
#             most_dissimilar_idx = np.argmax(avg_dissimilarity)
#
#             # Split the cluster based on dissimilarity
#             splinters = [cluster_data_indices[most_dissimilar_idx]]
#             remaining = cluster_data_indices.copy()
#             remaining.remove(splinters[0])
#
#             while remaining:
#                 split = False
#                 for j in remaining[:]:  # Iterate over a copy of remaining
#                     splinter_distances = diss_matrix[j, splinters]
#                     remaining_distances = diss_matrix[j, remaining]
#                     if np.mean(splinter_distances) <= np.mean(remaining_distances):
#                         splinters.append(j)
#                         remaining.remove(j)
#                         split = True
#                 if not split:
#                     break
#
#             # Record merge information
#             merge_list.append((max_cluster_idx, len(clusters), np.mean(diss_matrix[splinters][:, remaining])))
#
#             # Update clusters
#             del clusters[max_cluster_idx]
#             clusters.append(splinters)
#             clusters.append(remaining)
#
#             # Stop if target clusters are achieved
#             if self.stop_at_k and len(clusters) >= self.stop_at_k:
#                 break
#
#         self.clustering_result = {
#             'merge': merge_list,
#             'cluster_labels': np.zeros(n, dtype=int)
#         }
#
#         # Assign cluster labels
#         for i, cluster in enumerate(clusters):
#             self.clustering_result['cluster_labels'][cluster] = i
#
#         return self.clustering_result
#
#     def plot_clusters(self):
#         if self.clustering_result is not None:
#             labels = self.clustering_result['cluster_labels']
#             plt.figure(figsize=(10, 7))
#             plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
#             plt.title("DIANA Clustering Results")
#             plt.xlabel("Feature 1")
#             plt.ylabel("Feature 2")
#             plt.colorbar(label='Cluster Label')
#             plt.show()
#         else:
#             print("No clustering result available.")
#
#     def plot_dendrogram(self):
#         if self.clustering_result is not None:
#             merge = np.array(self.clustering_result['merge'])
#             fig, ax = plt.subplots(figsize=(10, 7))
#             sch.dendrogram(merge, labels=range(len(merge) + 1))
#             plt.title("Dendrogram for DIANA Clustering")
#             plt.xlabel("Sample index")
#             plt.ylabel("Distance")
#         return fig
