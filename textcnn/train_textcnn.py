import pandas as pd
import numpy as np
import pickle
import time
import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
import torch.nn.functional as F
print("================ HUẤN LUYỆN MÔ HÌNH TEXTCNN (PYTORCH) ================\n")

# 1. Cấu hình Device (Ưu tiên dùng GPU nếu có)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang sử dụng thiết bị: {device}")

# =====================================================================
# PHẦN 1: CHUẨN BỊ DỮ LIỆU & TỪ ĐIỂN
# =====================================================================
print("\nĐang tải dữ liệu...")
df_train = pd.read_csv("./preprocess/train_textcnn_phobert.csv")
df_test = pd.read_csv("./preprocess/test_textcnn_phobert.csv")

# --- ĐOẠN CODE SỬA LỖI ĐƯỢC THÊM VÀO ĐÂY ---
# Nếu nhãn trong file CSV là 1, 2, 3, 4, 5 thì trừ đi 1 để thành 0, 1, 2, 3, 4
if df_train['label'].max() == 5:
    print("⚠️ Cảnh báo: Phát hiện nhãn lệch (1-5). Đang tự động chuẩn hóa về (0-4)...")
    df_train['label'] = df_train['label'] - 1
    df_test['label'] = df_test['label'] - 1

# Đảm bảo không có nhãn nào nằm ngoài khoảng 0 -> 4
df_train = df_train[(df_train['label'] >= 0) & (df_train['label'] <= 4)]
df_test = df_test[(df_test['label'] >= 0) & (df_test['label'] <= 4)]
# ------------------------------------------

# Tải Label Encoder và Class Weights
with open("./preprocess/utils_and_weights.pkl", "rb") as f:
    utils = pickle.load(f)
le = utils['label_encoder']
target_names = le.classes_
class_weights_idx = utils['class_weights']

# Chuyển Class Weights thành Tensor của PyTorch
weights_list = [class_weights_idx[i] for i in range(len(le.classes_))]
weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(device)

print("\nĐang xây dựng từ điển (Vocabulary)...")
# Tự động tạo từ điển từ tập Train
word2idx = {'<pad>': 0, '<unk>': 1}
for text in df_train['text']:
    for word in str(text).split():
        if word not in word2idx:
            word2idx[word] = len(word2idx)

MAX_LEN = 100 # Chiều dài tối đa của một câu bình luận
VOCAB_SIZE = len(word2idx)
print(f"Kích thước từ điển: {VOCAB_SIZE} từ.")

# Hàm chuyển câu văn thành mảng các ID số
def encode_text(text):
    words = str(text).split()
    encoded = [word2idx.get(w, word2idx['<unk>']) for w in words]
    # Cắt ngắn nếu quá dài, thêm pad nếu quá ngắn
    if len(encoded) < MAX_LEN:
        encoded += [word2idx['<pad>']] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]
    return encoded

# Dataset dành cho PyTorch
class SentimentDataset(Dataset):
    def __init__(self, df):
        self.texts = torch.tensor([encode_text(t) for t in df['text']], dtype=torch.long)
        self.labels = torch.tensor(df['label'].values, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_dataset = SentimentDataset(df_train)
test_dataset = SentimentDataset(df_test)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================================================================
# PHẦN 2: ĐỊNH NGHĨA KIẾN TRÚC MẠNG TEXTCNN
# =====================================================================
class TextCNN(torch_nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters, dropout):
        super().__init__()
        self.embedding = torch_nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Tạo nhiều lớp Conv1d với các kernel_size khác nhau (tương đương n-grams)
        self.convs = torch_nn.ModuleList([
            torch_nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.fc = torch_nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = torch_nn.Dropout(dropout)

    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text) # shape: [batch_size, seq_len, embed_dim]
        # Chuyển vị trí để phù hợp với input của Conv1d trong PyTorch
        embedded = embedded.permute(0, 2, 1) # shape: [batch_size, embed_dim, seq_len]
        
        # Áp dụng Conv -> ReLU -> MaxPool
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Ghép nối các đặc trưng
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# Khởi tạo tham số mô hình
EMBED_DIM = 300
FILTER_SIZES = [2, 3, 4] # Bắt các cụm 2 từ, 3 từ, 4 từ đi liền nhau
NUM_FILTERS = 100
DROPOUT = 0.5
NUM_CLASSES = len(target_names)

model = TextCNN(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES, FILTER_SIZES, NUM_FILTERS, DROPOUT).to(device)

# =====================================================================
# PHẦN 3: HUẤN LUYỆN MÔ HÌNH (TRAINING LOOP)
# =====================================================================
# TRUYỀN CLASS WEIGHTS VÀO ĐÂY ĐỂ XỬ LÝ MẤT CÂN BẰNG
criterion = torch_nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10
print("\nBắt đầu huấn luyện...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_texts, batch_labels in train_loader:
        batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_texts)
        loss = criterion(predictions, batch_labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch: {epoch+1:02} | Train Loss: {epoch_loss/len(train_loader):.4f}")

print(f"\n✅ Huấn luyện xong! Thời gian: {time.time() - start_time:.2f} giây.")

# =====================================================================
# PHẦN 4: ĐÁNH GIÁ MÔ HÌNH (EVALUATION)
# =====================================================================
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, matthews_corrcoef, f1_score # Đã thêm f1_score

model.eval()
all_preds = []
all_labels = []
all_probs = [] # BỔ SUNG: List để lưu xác suất dự đoán

print("\nĐang đánh giá trên tập Test...")
with torch.no_grad():
    for batch_texts, batch_labels in test_loader:
        batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
        predictions = model(batch_texts) # Đây là logits
        
        # BỔ SUNG: Dùng Softmax để tính xác suất cho từng class
        probs = F.softmax(predictions, dim=1)
        all_probs.extend(probs.cpu().numpy())
        
        # Lấy nhãn có xác suất cao nhất
        preds = torch.argmax(predictions, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

# In báo cáo phân loại và lưu ra CSV
print("\n================ KẾT QUẢ ĐÁNH GIÁ CHI TIẾT ================\n")
report_dict = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
print("1. Báo Cáo Phân Loại (Classification Report):")
print(report_df.round(4))
report_df.to_csv("./textcnn/textcnn_classification_report.csv")

# ================= TÍNH TOÁN CÁC CHỈ SỐ =================
acc = accuracy_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)
f1_macro = f1_score(all_labels, all_preds, average='macro')
f1_weighted = f1_score(all_labels, all_preds, average='weighted')

# BỔ SUNG: Tính Log Loss
logloss = log_loss(all_labels, all_probs) 

print("\n2. Các Chỉ Số Tổng Thể (Global Metrics):")
print(f"- Accuracy (Độ chính xác):          {acc:.4f} ({acc*100:.2f}%)")
print(f"- F1-Score (Macro Average):         {f1_macro:.4f}")
print(f"- F1-Score (Weighted Average):      {f1_weighted:.4f}")
print(f"- MCC (Hệ số tương quan Matthews):  {mcc:.4f}")
print(f"- Log Loss:                         {logloss:.4f}") # BỔ SUNG: In ra màn hình
# ================= VẼ BIỂU ĐỒ TRỰC QUAN =================
print("\nĐang tạo Confusion Matrix...")
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
# Dùng tone màu khác (Oranges) để dễ phân biệt với SVM (Blues) trong báo cáo
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=target_names, yticklabels=target_names)
plt.title(f'Confusion Matrix - TextCNN Model\nAccuracy: {acc*100:.2f}% | F1-Macro: {f1_macro:.4f} | MCC: {mcc:.4f}')
plt.ylabel('Nhãn thực tế (True Label)')
plt.xlabel('Nhãn dự đoán (Predicted Label)')
plt.tight_layout()
plt.savefig('./textcnn/textcnn_confusion_matrix.png')

print("✅ Đã lưu hình ảnh biểu đồ vào 'textcnn_confusion_matrix.png'")
print("✅ Đã xuất báo cáo chi tiết ra file 'textcnn_classification_report.csv'")

# Lưu trọng số mô hình
torch.save(model.state_dict(), "./textcnn_model.pth")
print("✅ Đã lưu trọng số mô hình vào 'textcnn_model.pth'")