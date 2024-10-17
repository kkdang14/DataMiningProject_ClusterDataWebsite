import io
import os
import base64
import numpy as np
import pandas as pd
from flask import send_file
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import pdist, squareform
# import networkx as nx
from waitress import serve
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import OneHotEncoder
from flask import Flask, request, render_template, send_from_directory
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

app = Flask(__name__)

matplotlib.use('Agg')

# Đường dẫn thư mục lưu trữ file trên server
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Divisive clustering algorithm and tracking the tree structure
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

# Function to plot divisive clustering tree using networkx
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


def check_and_normalize_data(df):
    # Kiểm tra các cột không phải là số
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns

    if len(categorical_cols) > 0:
        # Nếu có cột phân loại, thực hiện One-Hot Encoding
        print(f"Các cột cần One-Hot Encoding: {list(categorical_cols)}")

        # Thực hiện One-Hot Encoding
        encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' để tránh multicollinearity
        encoded_data = encoder.fit_transform(df[categorical_cols])

        # Biến đổi kết quả từ One-Hot Encoding thành DataFrame
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

        # Kết hợp lại với các cột số đã có sẵn
        df = pd.concat([df.select_dtypes(include=['float64', 'int64']), encoded_df], axis=1)
        print(f"Sau khi chuẩn hóa, dữ liệu có các cột: {df.columns}")

    else:
        print("Dữ liệu đã đồng nhất, không cần One-Hot Encoding.")

    return df

# Trang chủ
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'static/favicon.ico', mimetype='image/vnd.microsoft.icon')

# Xử lý upload file và clustering
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    method = request.form.get('method')
    num_clusters = int(request.form.get('num_clusters', 3))  # Số cụm do người dùng nhập

    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        # Đọc dữ liệu từ file CSV hoặc Excel
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Kiểm tra và chuẩn hóa dữ liệu
        df = check_and_normalize_data(df)

        # Kiểm tra cột dữ liệu số
        numeric_data = df.select_dtypes(include=['float64', 'int64'])
        if numeric_data.empty:
            return 'Dataset không có cột số để gom nhóm.'

        # Kiểm tra số lượng cụm có hợp lý không
        if num_clusters > len(numeric_data):
            return f'Số lượng cụm ({num_clusters}) vượt quá số điểm dữ liệu ({len(numeric_data)}).'


        # Chọn phương pháp bottom-up hoặc top-down
        if method == 'bottom-up':
            # Apply Agglomerative Clustering for num_clusters
            clustering = AgglomerativeClustering(n_clusters=num_clusters)
            Z = linkage(numeric_data, 'ward')

            df['Cluster'] = clustering.fit_predict(numeric_data)

            method_name = "Bottom-Up (Agglomerative)"

            # Get the maximum distance at which to cut the dendrogram for num_clusters
            max_d = Z[-num_clusters, 2]

            # Plot the dendrogram
            plt.figure(figsize=(12, 8))
            plt.title(f"{method_name} Clustering Dendrogram")

            # Plot the dendrogram with a color threshold at max_d
            dendrogram_data = dendrogram(Z, color_threshold=max_d)
            print(dendrogram_data)

            # Horizontal line to represent the maximum distance (cut point)
            plt.axhline(y=max_d+1 , color='r', linestyle='-', linewidth=2)

        elif method == 'top-down':
            # Simulate divisive clustering with recursive KMeans
            df['Cluster'] = 0  # Start with one cluster
            ## Function for recursive KMeans with a stopping condition
            # Run the recursive KMeans
            df['Cluster'], cluster_tree = DiviseClustering(numeric_data.values, df['Cluster'], num_clusters)
            unique_labels = np.unique(df['Cluster'])
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            df['Cluster'] = df['Cluster'].map(label_mapping)
            method_name = "Top-Down (Recursive KMeans)"

            fig = plot_tree_dendrogram(cluster_tree, num_clusters)
            
        # Lưu biểu đồ dendrogram vào bộ nhớ
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        # Lưu biểu đồ dendrogram vào file
        plot_filename = os.path.join(UPLOAD_FOLDER, 'dendrogram.png')
        plt.savefig(plot_filename)  # Lưu vào file dendrogram.png


        # Lưu kết quả phân cụm vào file CSV
        result_filename = os.path.join(UPLOAD_FOLDER, 'clustering_result.csv')
        df.to_csv(result_filename, index=False)  # Lưu kết quả vào file CSV

        # Trả về kết quả dưới dạng HTML với nút tải xuống
        return render_template('result.html', table=df.to_html(),
                               method=method_name,
                               plot_url='/download/dendrogram',
                               result_url='/download/result')
    return 'File không hợp lệ. Vui lòng upload file CSV hoặc Excel.'

@app.route('/download/<file_type>')
def download_file(file_type):
    if file_type == 'dendrogram':
        file_path = os.path.join(UPLOAD_FOLDER, 'dendrogram.png')
        file_name = 'dendrogram.png'
    elif file_type == 'result':
        file_path = os.path.join(UPLOAD_FOLDER, 'clustering_result.csv')
        file_name = 'clustering_result.csv'
    else:
        return 'File không tồn tại.'

        # Set 'as_attachment=True' to trigger download dialog in the browser
    return send_file(file_path, as_attachment=True, download_name=file_name)


if __name__ == '__main__':
    # app.run(debug=False, host='0.0.0.0')
    app.run(debug=True, port=3001)
    pass