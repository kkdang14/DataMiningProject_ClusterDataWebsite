from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
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