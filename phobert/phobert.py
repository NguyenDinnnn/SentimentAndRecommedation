import os
import json
import torch
import torch.nn as nn
import re
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from underthesea import word_tokenize

# 1. Định nghĩa cấu trúc Model Hybrid tương thích với trọng số đã train
class PhoBertHybridModel(nn.Module):
    def __init__(self, num_classes=5):
        super(PhoBertHybridModel, self).__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(768 + 1, 256), 
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes) 
        )

    def forward(self, input_ids, attention_mask, star_feature):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output 
        combined_features = torch.cat((pooled_output, star_feature), dim=1) 
        logits = self.classifier(combined_features)
        return logits

# 2. Khai báo thông tin Repo Hugging Face và thiết bị chạy
REPO_ID = "Ariesnguyen12/phobert-rated"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Đang tải Tokenizer và Trọng số từ Hugging Face Hub...")

# Tải Tokenizer trực tiếp
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

# Tải file trọng số vật lý (Hugging Face tự động lưu vào bộ nhớ cache hệ thống)
weights_path = hf_hub_download(repo_id=REPO_ID, filename="pytorch_model.bin")

# Tải file ánh xạ nhãn đầu ra
label_map_path = hf_hub_download(repo_id=REPO_ID, filename="label_map.json")
with open(label_map_path, 'r', encoding='utf-8') as f:
    label_dict = json.load(f)

# 3. Khởi tạo và nạp trọng số mô hình
model = PhoBertHybridModel(num_classes=5)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval() # Chuyển sang chế độ đánh giá/suy luận

print("🎉 Nạp mô hình thành công! Sẵn sàng dự đoán.")

# =================================================================
# 4. HÀM DỰ ĐOÁN THỬ NGHIỆM (Dùng để kiểm tra tính năng)
# =================================================================
def predict_sentiment(text, star_rating):
    # Tiền xử lý văn bản cơ bản (Chuẩn hóa chuẩn underthesea)
    text = text.lower()
    text = re.sub(r'[^\w\s_]', ' ', text)
    text = word_tokenize(text, format="text")
    
    # Chuẩn hóa đặc trưng sao (tương tự lúc huấn luyện)
    star_feature = torch.tensor([[float(star_rating) / 5.0]], dtype=torch.float).to(device)
    
    # Tokenize câu đầu vào
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=120,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Dự đoán đầu ra không tính gradient
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, star_feature=star_feature)
        _, pred_idx = torch.max(logits, dim=1)
    
    # Ánh xạ từ chỉ số sang tên nhãn tiếng Việt cụ thể
    predicted_label = label_dict[str(pred_idx.item())]
    return predicted_label

# Chạy thử nghiệm mẫu
sample_text = "Sản phẩm dùng rất mượt mà, giao hàng lại nhanh nữa."
sample_star = 5
result = predict_sentiment(sample_text, sample_star)
print(f"\n[Thử nghiệm] Văn bản: '{sample_text}' | Số sao: {sample_star}")
print(f"👉 Kết quả phân loại cảm xúc: {result}")