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

# Function to plot dendrogram like AGNES tree
def plot_tree_dendrogram(tree, num_clusters):
    fig, ax = plt.subplots(figsize=(12, 8))
    x_coords = {}
    y_levels = {}

    # Color map để mỗi level có màu khác nhau
    colors = list(mcolors.TABLEAU_COLORS.values())

    for (depth, parent, child1, child2) in tree:
        if parent not in x_coords:
            x_coords[parent] = parent

        # Tạo khoảng cách cho các nhánh không bị trùng
        x_coords[child1] = x_coords[parent] - 1
        x_coords[child2] = x_coords[parent] + 1

        # Chọn màu theo depth
        color = colors[depth % len(colors)]

        # Vẽ đường ngang (để hiển thị độ sâu phân cụm trên trục y)
        ax.plot([x_coords[child1], x_coords[child2]], [depth, depth], color=color, lw=2)

        # Vẽ đường dọc xuống các nhánh
        ax.plot([x_coords[child1], x_coords[child1]], [depth, depth + 1], color=color, lw=2)
        ax.plot([x_coords[child2], x_coords[child2]], [depth, depth + 1], color=color, lw=2)

        # Ghi chú các cụm tại điểm phân nhánh
        ax.text(x_coords[child1], depth + 1.05, f'Cluster {child1}', verticalalignment='center', color=color, fontsize=10)
        ax.text(x_coords[child2], depth + 1.05, f'Cluster {child2}', verticalalignment='center', color=color, fontsize=10)

    # Cài đặt tiêu đề và trục
    ax.set_title("Divisive Clustering Dendrogram (Inverted Clustering)", fontsize=16)
    ax.set_ylabel("Depth of Clustering", fontsize=14)
    ax.set_xlabel("Cluster ID", fontsize=14)
    ax.invert_yaxis()
    # Thiết lập phạm vi trục x để không bị cắt bớt
    ax.set_xlim(min(x_coords.values()) - 1, max(x_coords.values()) + 1)
    return fig