import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_divisive_clustering(tree, num_clusters):
    fig, ax = plt.subplots(figsize=(10, 7))
    x_coords = {}

    colors = list(mcolors.TABLEAU_COLORS.values())

    for (depth, parent, child1, child2, _, _) in tree:
        if parent not in x_coords:
            x_coords[parent] = parent

        # Tạo ra khoảng cách cho các nhánh để tránh chồng chéo
        x_coords[child1] = x_coords[parent] - 1
        x_coords[child2] = x_coords[parent] + 1

        # Chọn màu dựa trên độ sâu
        color = colors[depth % len(colors)]

        # Vẽ các đường ngang để hiển thị các cụm
        ax.plot([depth, depth], [x_coords[child1], x_coords[child2]], color=color, lw=2)

        # Vẽ các đường thẳng đứng kết nối với các cụm
        ax.plot([depth + 1, depth], [x_coords[child1], x_coords[child1]], color=color, lw=2)
        ax.plot([depth + 1, depth], [x_coords[child2], x_coords[child2]], color=color, lw=2)

        # Tạo nhãn cho cụm
        ax.text(depth + 1.1, x_coords[child1], f'Cluster {int(child1)}', verticalalignment='center', color=color,
                fontsize=10)
        ax.text(depth + 1.1, x_coords[child2], f'Cluster {int(child2)}', verticalalignment='center', color=color,
                fontsize=10)

    ax.set_title("Inverted Divisive Clustering Dendrogram", fontsize=16)
    ax.set_xlabel("Depth of Clustering", fontsize=14)  # X-axis is depth
    ax.set_ylabel("Cluster ID", fontsize=14)  # Y-axis is cluster ID

    # Đặt giới hạn trục y để tránh bị cắt
    ax.set_ylim(min(x_coords.values()) - 1, max(x_coords.values()) + 1)
    return fig

