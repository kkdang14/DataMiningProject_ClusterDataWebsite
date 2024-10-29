from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

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