from flask import Flask, request, render_template
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def custom_divisive_clustering(df, k_clusters):
    data = df.values
    clusters = [(data, 0)]  # Tuples of (data points, cluster_id)
    Z = []  # Linkage matrix
    cluster_id = 0

    while len(clusters) < k_clusters:
        new_clusters = []
        for points, cid in clusters:
            if len(points) <= 1:
                continue  # Không chia nếu chỉ có một điểm

            # Chia thành 2 cụm
            n_clusters = min(2, k_clusters - len(new_clusters))

            # Sử dụng KMeans để chia cụm hiện tại thành n_clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(points)
            labels = kmeans.labels_

            unique_labels = np.unique(labels)

            # Tạo các cụm con mới
            for i in unique_labels:
                cluster_points = points[labels == i]
                if len(cluster_points) > 0:
                    # Tính toán khoảng cách giữa các cụm
                    dist = np.linalg.norm(kmeans.cluster_centers_[i] - np.mean(cluster_points, axis=0))

                    # Đảm bảo ID cụm là duy nhất
                    Z.append([cid, cluster_id, dist, len(cluster_points)])
                    new_clusters.append((cluster_points, cluster_id))
                    cluster_id += 1

            # Kiểm tra nếu đã đủ số cụm
            if len(new_clusters) >= k_clusters:
                break

        clusters.extend(new_clusters)

    return np.array(Z)


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    algorithm = request.form['algorithm']
    n = int(request.form['n-cluster'])
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)

    # Loại bỏ hàng trùng lặp
    df = df.drop_duplicates()

    # Kiểm tra dữ liệu
    if df.isnull().values.any():
        return "Dữ liệu chứa giá trị bị thiếu. Vui lòng kiểm tra lại."

    # Đảm bảo dữ liệu là số
    df = df.select_dtypes(include=[np.number])
    if df.empty:
        return "Dữ liệu không chứa đặc trưng số. Vui lòng kiểm tra lại."

    # Perform hierarchical clustering
    if algorithm == 'bottom-up':
        Z = linkage(df, 'ward')
    elif algorithm == 'top-down':
        Z = custom_divisive_clustering(df, n)

    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(Z, leaf_rotation=90, leaf_font_size=12, color_threshold=1)

    # Save plot to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return render_template('result.html', image=image_base64)


if __name__ == '__main__':
    app.run(debug=True)

