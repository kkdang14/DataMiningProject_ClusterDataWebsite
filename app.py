import io
import os
import base64
import secrets

import numpy as np
import pandas as pd
from flask import send_file, session
import matplotlib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for
from sklearn.cluster import AgglomerativeClustering
from controller.Divise import DiviseClustering, plot_tree_dendrogram
from controller.preprocessing import check_and_normalize_data

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

@app.route('/cluster', methods=['POST'])
def cluster():
    file = request.files['file']
    method = request.form.get('method')
    num_clusters = int(request.form.get('num_clusters', 3))
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        # Đọc dữ liệu từ file CSV hoặc Excel
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        df = check_and_normalize_data(df)

        # Kiểm tra cột dữ liệu số
        numeric_data = df.select_dtypes(include=['float64', 'int64'])

        if numeric_data.empty:
            flash('Dataset không có cột số để gom nhóm.', 'warning')
            return redirect(url_for('home'))

        if num_clusters > len(df):
            flash(f'The number of clusters ({num_clusters}) exceeds the number of data points ({len(df)}).', 'warning')
            return redirect(url_for('home'))

        if method == 'bottom-up':
            # Apply Agglomerative Clustering for num_clusters
            clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean')
            Z = linkage(df, 'ward')

            df['Cluster'] = clustering.fit_predict(df)

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

            df['Cluster'] = 0  # Start with one cluster
            df['Cluster'], cluster_tree = DiviseClustering(df.values, df['Cluster'], num_clusters)
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
    # app.run(debug=True)
    # app.run(debug=False, host='0.0.0.0')
    # app.run(debug=True, port=3001)
    pass