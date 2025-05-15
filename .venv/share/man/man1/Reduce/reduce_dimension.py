import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import joblib
import shutil
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

# === Cấu hình ===
file_path = "Lables.xlsx"
image_folder = "Segmented"
image_size = (128, 128)
output_file = "reduced_data_pca.joblib"
output_csv = "processed_labels.csv"
output_npy_folder = "reduce_segment"
output_tif_folder = "reduced_segment_tif"

# === Đọc file Excel ===
df = pd.read_excel(file_path)

# Các cột tên
image_column = "Tên ảnh"
hand_label_column = "Nhãn bàn tay"
thumb_label_column = "Nhãn ngón tay cái"

if image_column not in df.columns:
    raise KeyError(f"Cột '{image_column}' không tồn tại trong file Excel!")

image_data = []
hand_labels = []
thumb_labels = []
valid_entries = []

# Danh sách file trong thư mục ảnh
all_images = os.listdir(image_folder)
print(f"Tìm thấy {len(all_images)} file trong thư mục '{image_folder}'.")

# Lọc và xử lý ảnh từ 4157 đến 7115
for index, row in df.iterrows():
    excel_name = row[image_column]
    base_name = os.path.splitext(excel_name)[0]

    try:
        number = int(base_name)
    except ValueError:
        print(f"Bỏ qua tên ảnh không hợp lệ: {base_name}")
        continue

    if not (4157 <= number <= 7115):
        continue

    # Kiểm tra file ảnh có tồn tại không
    tif_path = os.path.join(image_folder, f"{number}_Segmented.tif")
    png_path = os.path.join(image_folder, f"{number}_Segmented.png")

    if os.path.exists(tif_path):
        img_path = tif_path
    elif os.path.exists(png_path):
        img_path = png_path
    else:
        print(f"Không tìm thấy ảnh cho ID {number}")
        continue

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Không thể tải ảnh: {img_path}")
        continue

    img = cv2.resize(img, image_size)
    image_data.append(img.flatten())
    hand_labels.append(row[hand_label_column])
    thumb_labels.append(row.get(thumb_label_column, ""))
    valid_entries.append({
        "image_id": number,
        "hand_label": row[hand_label_column],
        "thumb_label": row.get(thumb_label_column, "")
    })

# Kiểm tra dữ liệu trước khi tiếp tục
if len(image_data) == 0:
    raise ValueError("Không có ảnh hợp lệ. Kiểm tra lại thư mục ảnh và file Excel.")

X = np.array(image_data)
y_hand = np.array(hand_labels)

# Mã hóa nhãn bàn tay (L/R) thành 0/1
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_hand)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA - Giảm chiều
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"Số lượng ảnh đã xử lý: {len(image_data)}")
print(f"Số nhãn bàn tay duy nhất: {len(set(hand_labels))}")
print(f"Số nhãn ngón tay cái duy nhất: {len(set(thumb_labels))}")
print(f"Số thành phần PCA giữ lại: {X_pca.shape[1]}")

# Lưu dữ liệu đã giảm chiều
joblib.dump((X_pca, y_encoded), output_file)
print(f"Dữ liệu PCA đã được lưu vào: {output_file}")

# Lưu thông tin nhãn vào file CSV
df_out = pd.DataFrame(valid_entries)
df_out.to_csv(output_csv, index=False)
print(f"Thông tin nhãn đã được lưu vào: {output_csv}")

# === Tạo thư mục để lưu ảnh gốc & ảnh sau PCA ===
os.makedirs(output_npy_folder, exist_ok=True)
os.makedirs(output_tif_folder, exist_ok=True)

# === Sao chép ảnh gốc vào thư mục reduced_segment_tif ===
for entry in valid_entries:
    original_tif = os.path.join(image_folder, f"{entry['image_id']}_Segmented.tif")
    original_png = os.path.join(image_folder, f"{entry['image_id']}_Segmented.png")

    if os.path.exists(original_tif):
        shutil.copy(original_tif, os.path.join(output_tif_folder, f"{entry['image_id']}_Segmented.tif"))
    elif os.path.exists(original_png):
        shutil.copy(original_png, os.path.join(output_tif_folder, f"{entry['image_id']}_Segmented.png"))
    else:
        print(f"Không tìm thấy ảnh gốc cho ID {entry['image_id']}")

print(f"Tất cả ảnh gốc đã được sao chép vào thư mục '{output_tif_folder}'.")

# === Lưu dữ liệu PCA dưới dạng .npy ===
for idx, img_data in enumerate(X_pca):
    img_rescaled = (img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255
    npy_save_path = os.path.join(output_npy_folder, f"{valid_entries[idx]['image_id']}_Reduced.npy")
    np.save(npy_save_path, img_rescaled)

print(f"Tất cả ảnh sau giảm chiều đã được lưu vào thư mục '{output_npy_folder}' dưới dạng .npy.")

# === Trực quan hóa PCA ===
if X_pca.shape[1] >= 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='coolwarm', alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization of X-ray Hand Images")
    plt.colorbar(label="Hand Label (0 = L, 1 = R)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Không đủ thành phần PCA để trực quan hóa.")
