import io
import os
import base64
import secrets
import tempfile
import numpy as np
import pandas as pd
from flask import send_file, session
import matplotlib
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from scipy.cluster.hierarchy import linkage, dendrogram
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from controller.Divisive import DivisiveClustering
from controller.preprocessing import check_and_normalize_data
from controller.FindBestCluster import find_best_number_of_cluster
from controller.Visualize import plot_divisive_clustering


app = Flask(__name__, template_folder="templates")
secret_key = secrets.token_hex(16)
app.secret_key = secret_key
matplotlib.use('Agg')

# Đường dẫn thư mục lưu trữ file trên server
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Trang chủ


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'static/favicon.ico', mimetype='image/vnd.microsoft.icon')
@app.route('/check', methods=['POST'])
def checking_cluster():
    file = request.files['file-checking']
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)  # Save the file to the server
        session['uploaded_file_path'] = file_path  # Save file path in session
    else:
        flash('Invalid file. Please upload a CSV or Excel file.', 'danger')
        return redirect(url_for('home'))

    # Read data from the saved file
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # Kiểm tra và chuẩn hóa dữ liệu
    df = check_and_normalize_data(df)

    # Kiểm tra dữ liệu số trong mỗi cột
    numeric_data = df.select_dtypes(include=['float64', 'int64'])
    if numeric_data.empty:
        flash('Dataset does not contain numeric columns for clustering.', 'warning')
        return redirect(url_for('home'))

    # Tìm số cụm tối ưu nhất cho dữ liệu
    best_clusters = find_best_number_of_cluster(df)

    flash(f'Optimal number of clusters determined to be {best_clusters}.', 'info')
    return redirect(url_for('home'))


@app.route('/cluster', methods=['POST'])
def cluster():
    file = session.get('uploaded_file_path')
    if 'file' in request.files and request.files['file']:
        # Nếu có file được upload từ chức năng gom cụm
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)  # Save the new file
        session['uploaded_file_path'] = file_path  # Cập nhật lại file trong session
    elif file:
        # Nếu không có file upload từ chức năng gom cụm, lấy file từ session checking clustering làm file thực hiện
        #gom cụm
        file_path = file
    else:
        flash("No file found. Please check or upload a dataset first.", 'warning')
        return redirect(url_for('home'))

    method = request.form.get('method')
    num_clusters = int(request.form.get('num_clusters', 3))

    # Read data from the saved file
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        flash('Dataset does not match the required file type', 'warning')
        return redirect(url_for('home'))

    # Xóa file trong session sau khi đã thực hiện đọc
    os.remove(file_path)

    # Kiểm tra và chuẩn hóa dữ liệu
    df = check_and_normalize_data(df)

    # Kiểm tra cột dữ liệu số
    numeric_data = df.select_dtypes(include=['float64', 'int64'])

    #Kiểm tra cột
    if numeric_data.empty:
        flash('Dataset is not column for clustering.', 'warning')
        return redirect(url_for('home'))

    # Kiểm tra số cụm so với điểm dữ liệu
    if num_clusters > len(df):
        flash(f'The number of clusters ({num_clusters}) exceeds the number of data points ({len(df)}).', 'warning')
        return redirect(url_for('home'))

    clustering_steps = []
    # Triển khai 2 chiến lược gom cụm
    if method == 'bottom-up':
        # Áp dụng thuật toán Agglomerative Clustering
        clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean')
        Z = linkage(df, 'ward')

        df['Cluster'] = clustering.fit_predict(df)

        method_name = "Bottom-Up (Agglomerative)"

        # Khởi tạo kích thước cụm cho từng điểm ban đầu
        cluster_sizes = {i: 1 for i in range(len(df))}

        # Thu thập các bước gom cụm
        for i in range(len(Z)):
            idx1, idx2, distance = Z[i, :3]

            # Tính kích thước của cụm mới
            cluster_size = cluster_sizes.get(int(idx1), 0) + cluster_sizes.get(int(idx2), 0)

            # Cập nhật kích thước cụm mới
            cluster_sizes[len(df) + i] = cluster_size

            # Lưu thông tin bước vào `clustering_steps`
            clustering_steps.append({
                "Step": i + 1,
                "Cluster 1": int(idx1),
                "Cluster 2": int(idx2),
                "Distance": distance,
                "New Cluster Size": cluster_size
            })

        # Lấy ra khoảng cách lớn nhất để cắt biểu đồ dendrogram tại vị trí chia cụm
        max_d = Z[-num_clusters, 2]

        # Trực quan hóa dendrogram
        plt.figure(figsize=(12, 8))
        plt.title(f"{method_name} Clustering Dendrogram")
        # Tạo ra và in ra dữ liệu của một dendrogram dựa trên ma trận liên kết Z
        dendrogram_data = dendrogram(Z, color_threshold=max_d)
        print(dendrogram_data)

        # Tạo đường cắt ngang, xác định và trực quan hóa ngưỡng phân cụm trong dendrogram
        plt.axhline(y=max_d+1 , color='r', linestyle='-', linewidth=2)

    elif method == 'top-down':
        df['Cluster'] = 0  # Bắt đầu với cụm không
        # Gọi hàm DivisiveClustering tự cài đặt
        df['Cluster'], cluster_tree = DivisiveClustering(df.values, df['Cluster'], num_clusters)
        unique_labels = np.unique(df['Cluster'])
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        df['Cluster'] = df['Cluster'].map(label_mapping)
        method_name = "Top-Down (Divisive Clustering)"

        print(cluster_tree)

        for step in cluster_tree:
            clustering_steps.append({
                "Step": str(step[0] + 1),
                "Cluster Id": step[1],
                "Cluster 1": step[2],
                "Cluster 2": step[3],
                "Distance": step[4],
                "New Cluster Size": step[5]
            })

        fig = plot_divisive_clustering(cluster_tree, num_clusters)

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
                            clustering_steps=clustering_steps,
                            plot_url='/download/dendrogram',
                            result_url='/download/result')

    flash('Invalid file type. Please upload a CSV or Excel file.', 'danger')
    return redirect(url_for('home'))
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
    app.run(debug=True, port=3001)
    # pass