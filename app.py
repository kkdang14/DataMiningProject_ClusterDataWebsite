import io
import os
import base64
import numpy as np
import pandas as pd
from flask import send_file
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from flask import Flask, request, render_template
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans

app = Flask(__name__)

# Đường dẫn thư mục lưu trữ file trên server
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Function to calculate SSE for a cluster
def calculate_sse(data, labels):
    sse = []
    for i in np.unique(labels):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            center = cluster_points.mean(axis=0)
            sse.append(np.sum((cluster_points - center) ** 2))
        else:
            sse.append(0)
    return np.array(sse)


# Recursive function to perform the divisive clustering and splitting
def top_down_clustering(data, num_clusters, cluster_id=0, labels=None, original_index=None):
    if data is None:
        raise ValueError("Data cannot be None in recursive_clustering function.")

    # Initialize labels and original indices if not provided
    if labels is None:
        labels = np.zeros(data.shape[0], dtype=int)
    if original_index is None:
        original_index = np.arange(data.shape[0])

    print(f"Recursive clustering called with {len(data)} data points and {num_clusters} clusters.")

    # Perform K-Means clustering with 2 clusters to split the cluster
    kmeans = KMeans(n_clusters=2)
    sub_labels = kmeans.fit_predict(data)

    # Calculate SSE for each cluster
    sse = calculate_sse(data, sub_labels)
    max_sse_index = np.argmax(sse)
    print(f"Cluster {max_sse_index} has the highest SSE: {sse[max_sse_index]}")


    # Assign new labels in the original labels array
    if len(labels) != len(original_index):
        raise ValueError(f"Length mismatch: labels {len(labels)} vs. original_index {len(original_index)}")

    labels[original_index[sub_labels == max_sse_index]] = cluster_id

    # If the current number of clusters is less than the target, split further
    if cluster_id + 1 < num_clusters - 1:
        for i in range(2):  # Loop through both sub-clusters
            cluster_points = data[sub_labels == i]
            new_index = original_index[sub_labels == i]
            print(f"Splitting cluster {i} with {len(cluster_points)} points")
            if len(cluster_points) > 0:
                top_down_clustering(cluster_points, num_clusters, cluster_id + i, labels, new_index)

    return labels


# Trang chủ
@app.route('/')
def home():
    return render_template('index.html')

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

        # Kiểm tra cột dữ liệu số
        numeric_data = df.select_dtypes(include=['float64', 'int64'])
        if numeric_data.empty:
            return 'Dataset không có cột số để gom nhóm.'

        # Kiểm tra số lượng cụm có hợp lý không
        if num_clusters > len(numeric_data):
            return f'Số lượng cụm ({num_clusters}) vượt quá số điểm dữ liệu ({len(numeric_data)}).'

        # Chọn phương pháp bottom-up hoặc top-down
        if method == 'bottom-up':
            Z = linkage(numeric_data, 'ward')
            clustering = AgglomerativeClustering(n_clusters=num_clusters)
            df['Cluster'] = clustering.fit_predict(numeric_data)
            method_name = "Bottom-Up (Agglomerative)"

            max_d = Z[-num_clusters, 2]
            # Vẽ biểu đồ Dendrogram cho Bottom-up
            plt.figure(figsize=(10, 7))
            plt.title(f"{method_name} Clustering Dendrogram")
            dendrogram(Z, color_threshold=max_d)
            plt.axhline(y=max_d, color='r', linestyle='--')

        elif method == 'top-down':
            # Áp dụng custom top-down divisive clustering
            try:
                Z = linkage(numeric_data, method='ward')
                labels = top_down_clustering(numeric_data.to_numpy(), num_clusters)
                df['Cluster'] = labels
                method_name = "Top-Down (Divisive)"
            except Exception as e:
                return f"Error in top-down clustering: {str(e)}"
            max_d = Z[-num_clusters, 2]
            # Plot the dendrogram manually
            plt.figure(figsize=(10, 7))
            plt.title(f"{method_name} Clustering Dendrogram")
            dendrogram(Z, color_threshold=max_d)
            plt.axhline(y=max_d, color='r', linestyle='--')

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
    elif file_type == 'result':
        file_path = os.path.join(UPLOAD_FOLDER, 'clustering_result.csv')
    else:
        return 'File không tồn tại.'

    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    # app.run(debug=False, host='0.0.0.0')
    app.run(debug=True)