import os
import pandas as pd
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cấu hình đường dẫn
CSV_PATH = 'Clustered_Lables.csv'
IMAGE_FOLDER = 'cluster_images'
OUTPUT_IMG_FOLDER = 'output_images'
OUTPUT_NPY_FOLDER = 'output_npy'

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(OUTPUT_IMG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_NPY_FOLDER, exist_ok=True)

# Đọc file CSV
df = pd.read_csv(CSV_PATH, encoding='iso-8859-1')

# Tách ID từ tên ảnh
df['id'] = df['Tên ?nh'].str.extract(r'(\d+)\.png')

# Danh sách đặc trưng và thông tin đi kèm
features = []
valid_ids = []
original_images = []

# Đọc và xử lý từng ảnh
for img_id in df['id']:
    img_filename = f"{img_id}_Segmented.tif"
    img_path = os.path.join(IMAGE_FOLDER, img_filename)

    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized = cv2.resize(img, (64, 64))
            flat = resized.flatten()
            features.append(flat)
            valid_ids.append(img_id)
            original_images.append(resized)
    else:
        print(f"⚠️ Không tìm thấy file: {img_path}")

# Kiểm tra nếu không có dữ liệu
if not features:
    raise ValueError("❌ Không có ảnh hợp lệ nào được xử lý. Kiểm tra lại thư mục ảnh và file CSV.")

# Giảm chiều với PCA
X = np.array(features)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Ghi vector PCA và ảnh đã resize tương ứng
for i, img_id in enumerate(valid_ids):
    # Ghi vector PCA
    npy_path = os.path.join(OUTPUT_NPY_FOLDER, f"{img_id}_pca.npy")
    np.save(npy_path, X_reduced[i])

    # Lưu ảnh gốc đã resize
    output_img_path = os.path.join(OUTPUT_IMG_FOLDER, f"{img_id}.png")
    cv2.imwrite(output_img_path, original_images[i])

# Tạo dataframe kết quả
result_df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
result_df['id'] = valid_ids

# Gộp với thông tin gốc từ CSV (bao gồm cột Cluster)
final_df = pd.merge(df, result_df, on='id', how='inner')

# Lưu lại
final_df.to_csv('reduced_data.csv', index=False)
print("✅ Đã lưu reduced_data.csv, vector PCA (.npy), và ảnh resized (.png).")

# === VẼ BIỂU ĐỒ PCA TỔNG PHÂN LOẠI THEO CỤM ===
plt.figure(figsize=(8, 6))
clusters = final_df['Cluster'].unique()
colors = plt.cm.get_cmap('tab10', len(clusters))

for idx, cluster in enumerate(clusters):
    cluster_data = final_df[final_df['Cluster'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'],
                label=f'Cluster {cluster}',
                color=colors(idx), s=30, alpha=0.7)

plt.title('Biểu đồ PCA theo cụm')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
