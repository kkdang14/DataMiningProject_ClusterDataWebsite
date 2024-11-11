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
            # Tính ma trận độ không tương đồng
            dissimilarity_matrix = pairwise_distances(cluster_data, metric='euclidean')

            # Tính độ không tương đồng trung bình
            avg_dissimilarity = dissimilarity_matrix.mean(axis=1)
            most_dissimilar_point = np.argmax(avg_dissimilarity)

            # Tách cụm
            new_labels = np.zeros(len(cluster_data))
            new_labels[most_dissimilar_point] = 1

            distances = dissimilarity_matrix[most_dissimilar_point]
            new_labels[distances >= np.median(distances)] = 1

            new_cluster_ids = new_labels + np.max(current_cluster) + 1
            current_cluster = current_cluster.copy()
            current_cluster[current_cluster == cluster_id] = new_cluster_ids

            tree.append((depth, cluster_id, np.max(new_cluster_ids), np.min(new_cluster_ids), np.median(distances), len(cluster_data)))

            # Kiểm tra nếu đã đạt đủ số lượng cụm
            if len(np.unique(current_cluster)) >= target_clusters:
                break

            # Phân tách đệ quy
            current_cluster, tree = DivisiveClustering(data, current_cluster, target_clusters, depth + 1, tree)

    return current_cluster, tree
