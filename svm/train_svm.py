import pickle
import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix, 
    matthews_corrcoef, 
    log_loss,
    f1_score # Đã thêm import f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

print("================ HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH SVM ================\n")

# 1. Tải dữ liệu đã tiền xử lý
print("Đang tải dữ liệu từ data_for_svm.pkl...")
with open("./preprocess/data_for_svm.pkl", "rb") as f:
    data_svm = pickle.load(f)

X_train = data_svm['X_train']
y_train = data_svm['y_train']
X_test = data_svm['X_test']
y_test = data_svm['y_test']

# Tải LabelEncoder để dịch nhãn
with open("./preprocess/utils_and_weights.pkl", "rb") as f:
    utils = pickle.load(f)
le = utils['label_encoder']
target_names = le.classes_

print(f"Kích thước tập Train (đã SMOTE): {X_train.shape}")
print(f"Kích thước tập Test: {X_test.shape}")

# 2. Khởi tạo và Huấn luyện mô hình SVM
print("\nĐang khởi tạo mô hình SVM (kernel='linear')...")
# Bật probability=True để tính được Log Loss
svm_model = SVC(kernel='linear', probability=True, random_state=42, verbose=True)

print("Đang huấn luyện mô hình (Vui lòng chờ)...")
start_time = time.time()
svm_model.fit(X_train, y_train)
end_time = time.time()
print(f"✅ Đã huấn luyện xong! Thời gian: {end_time - start_time:.2f} giây.")

# 3. Dự đoán trên tập Test
print("\nĐang thực hiện dự đoán trên tập Test...")
y_pred = svm_model.predict(X_test)
y_pred_proba = svm_model.predict_proba(X_test) # Dùng cho Log Loss

# =====================================================================
# 4. TÍNH TOÁN BỘ CHỈ SỐ ĐÁNH GIÁ CHI TIẾT
# =====================================================================
print("\n================ KẾT QUẢ ĐÁNH GIÁ CHI TIẾT ================\n")

# A. Báo cáo phân loại (Precision, Recall, F1 theo từng nhãn)
report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
print("1. Báo Cáo Phân Loại (Classification Report):")
print(report_df.round(4))
report_df.to_csv("./svm/svm_classification_report.csv")

# B. Các chỉ số tổng thể (Global Metrics)
acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)

# --- TÍNH TOÁN F1-SCORE TỔNG THỂ ---
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print("\n2. Các Chỉ Số Tổng Thể (Global Metrics):")
print(f"- Accuracy (Độ chính xác):          {acc:.4f} ({acc*100:.2f}%)")
print(f"- F1-Score (Macro Average):         {f1_macro:.4f} (Rất quan trọng để đánh giá độ cân bằng)")
print(f"- F1-Score (Weighted Average):      {f1_weighted:.4f}")
print(f"- MCC (Hệ số tương quan Matthews):  {mcc:.4f} (Càng gần 1 càng tốt)")
print(f"- Log Loss:                         {logloss:.4f} (Càng thấp càng tốt)")

# =====================================================================
# 5. VẼ BIỂU ĐỒ TRỰC QUAN
# =====================================================================
print("\nĐang tạo Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
# Đưa thêm F1-Macro lên tiêu đề biểu đồ cho trực quan
plt.title(f'Confusion Matrix - SVM Model\nAccuracy: {acc*100:.2f}% | F1-Macro: {f1_macro:.4f} | MCC: {mcc:.4f}')
plt.ylabel('Nhãn thực tế (True Label)')
plt.xlabel('Nhãn dự đoán (Predicted Label)')
plt.tight_layout()
plt.savefig('./svm/svm_confusion_matrix.png')
print("✅ Đã lưu hình ảnh biểu đồ vào './svm/svm_confusion_matrix.png'")
print("✅ Đã xuất báo cáo chi tiết ra file './svm/svm_classification_report.csv'")

# Lưu mô hình
with open("./svm/svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)