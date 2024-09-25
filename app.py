from flask import Flask, request, render_template
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import os
from flask import send_file
import base64
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering

app = Flask(__name__)


# Hàm thực hiện phân cụm đệ quy cho top-down
def top_down_clustering(data, num_clusters):
    clusters = [(data, 0)]  # Danh sách cụm ban đầu
    labels = np.zeros(data.shape[0], dtype=int)  # Nhãn cho các điểm dữ liệu
    current_cluster_label = 1  # Đánh dấu nhãn cụm

    while len(clusters) < num_clusters:
        # Nếu không có cụm nào để chia, dừng vòng lặp
        if len(clusters) == 0:
            break

        # Tìm cụm lớn nhất để chia
        largest_cluster_index = max(range(len(clusters)), key=lambda i: len(clusters[i][0]))
        largest_cluster_data = clusters[largest_cluster_index][0]

        # Nếu cụm lớn có ít hơn 2 điểm, không thể chia thêm
        if len(largest_cluster_data) < 2:
            break

        # Xóa cụm lớn nhất ra khỏi danh sách
        clusters.pop(largest_cluster_index)

        # Áp dụng KMeans để chia cụm thành 2 phần
        kmeans = KMeans(n_clusters=2)
        sub_clusters = kmeans.fit_predict(largest_cluster_data)

        # Gán nhãn cho các điểm
        labels[labels == 0] = current_cluster_label  # Nhãn cho cụm mới 1
        labels[labels == 1] = current_cluster_label + 1  # Nhãn cho cụm mới 2

        # Thêm hai cụm mới vào danh sách
        clusters.append((largest_cluster_data[sub_clusters == 0], current_cluster_label))
        clusters.append((largest_cluster_data[sub_clusters == 1], current_cluster_label + 1))

        current_cluster_label += 2  # Tăng nhãn cụm

    return labels

# Đường dẫn thư mục lưu trữ file trên server
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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

            # Vẽ biểu đồ Dendrogram cho Bottom-up
            plt.figure(figsize=(10, 7))
            plt.title(f"{method_name} Clustering Dendrogram")
            dendrogram(Z)

        elif method == 'top-down':
            # Áp dụng custom top-down divisive clustering
            labels = top_down_clustering(numeric_data.to_numpy(), num_clusters)
            df['Cluster'] = labels
            method_name = "Top-Down (Divisive)"

            # Vẽ Dendrogram cho Top-down bằng cách tính khoảng cách giữa các cụm
            dist_matrix = squareform(pdist(numeric_data, 'euclidean'))
            plt.figure(figsize=(10, 7))
            plt.title(f"{method_name} Clustering Dendrogram")
            dendrogram(dist_matrix)

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

    # Thêm hàm để tải file dendrogram
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
    app.run(debug=True)
