import pandas as pd
import re
from underthesea import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import numpy as np
import pickle

# =====================================================================
# PHẦN 1: TỪ ĐIỂN VÀ HÀM TIỀN XỬ LÝ
# =====================================================================
emoticon_dict = {
    "❤️": " yêu_thích ", "❤": " yêu_thích ", "😍": " tuyệt_vời ", "👍": " tốt ", 
    "😭": " khóc ", "😂": " cười_lớn ", "🤡": " lừa_đảo ", "😡": " tức_giận ",
    "=))": " cười_lớn ", ":v": " cười_nhếch_mép ", ":(": " buồn_bã ", "^^": " vui_vẻ ",
    "huhu": " khóc ", ":D": " cười_vui "
}

punctuation_dict = {
    "!": " nhấn_mạnh_cảm_xúc ", "?": " nghi_ngờ_cảm_xúc "
}

teencode_dict = {
    "sp": "sản_phẩm", "đc": "được", "dc": "được", "k": "không", "ko": "không",
    "kh": "không", "khg": "không", "rep": "trả_lời", "onl": "online",
    "auth": "chính_hãng", "fake": "giả_mạo", "gòi": "rồi", "qá": "quá",
    "đt": "điện_thoại", "ok": "đồng_ý", "r": "rồi", "nhaa": "nha", "nhaan": "nhận",
    "2day": "hai_dây"
}

stopwords = ["là", "và", "thì", "mà", "các", "của", "cho", "những", "có", "với", "như", "để"]

def text_preprocess(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    for emo, meaning in emoticon_dict.items(): text = text.replace(emo, meaning)
    text = re.sub(r'!+', punctuation_dict["!"], text)
    text = re.sub(r'\?+', punctuation_dict["?"], text)
    text = re.sub(r'[^\w\s_]', ' ', text)
    words = text.split()
    words = [teencode_dict.get(w, w) for w in words]
    text = " ".join(words)
    text = word_tokenize(text, format="text")
    words = text.split()
    words = [w for w in words if w not in stopwords]
    text = " ".join(words)
    return text.strip()

# =====================================================================
# PHẦN 2: ĐỌC DỮ LIỆU dataset.csv (Dữ liệu gốc) VÀ CHUYỂN ĐỔI NHÃN
# =====================================================================
print("Đang đọc file dataset gốc (dataset.csv)...")
df = pd.read_csv("./dataset/dataset.csv")

# Đảm bảo không có cột thừa
if 'Unnamed: 3' in df.columns:
    df = df.drop(columns=['Unnamed: 3'])

# ĐÃ SỬA: Chuyển đổi nhãn CHÍNH XÁC theo yêu cầu của bạn
mapping_rate = {
    5: 'Tuyệt vời',
    4: 'Tốt',
    3: 'Bình thường',
    2: 'Tệ',
    1: 'Rất tệ'
}
df['label_5_classes'] = df['rate'].map(mapping_rate)
df = df.dropna(subset=['comment', 'label_5_classes'])

# TIỀN XỬ LÝ TEXT
print("Đang tiến hành tiền xử lý văn bản, vui lòng chờ...")
df['comment_preprocessed'] = df['comment'].apply(text_preprocess)
df = df.dropna(subset=['comment_preprocessed'])
df = df[df['comment_preprocessed'].str.strip() != ''] 

print("\n--- Phân bố dữ liệu 5 nhãn BAN ĐẦU ---")
print(df['label_5_classes'].value_counts())

# =====================================================================
# PHẦN 3: LƯU TEXT ĐÃ TIỀN XỬ LÝ CHO PHOBERT & TEXTCNN
# =====================================================================
print("\nĐang mã hóa nhãn...")
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label_5_classes'])
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"Bản đồ nhãn (Máy tính hiểu): {label_mapping}")

X_text = df['comment_preprocessed'].values
y_encoded = df['label_encoded'].values

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# =====================================================================
# PHẦN 4: VECTOR HÓA BẰNG TF-IDF DÀNH RIÊNG CHO SVM
# =====================================================================
print("\nĐang chuyển hóa văn bản thành TF-IDF (Cho SVM)...")
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 3))
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# =====================================================================
# PHẦN 5: XỬ LÝ MẤT CÂN BẰNG
# =====================================================================
print("\n================ CÁC PHƯƠNG PHÁP XỬ LÝ MẤT CÂN BẰNG ================")

print("\n1. Đang áp dụng SMOTE cho SVM...")
smote = SMOTE(random_state=42)
X_train_tfidf_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

print("\n2. Đang tính toán Class Weights cho TextCNN & PhoBERT...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(zip(le.classes_, class_weights))
class_weights_idx = dict(enumerate(class_weights)) 

print("Trọng số phân bổ cho từng nhãn:")
for label, weight in class_weights_dict.items():
    print(f"- {label}: {weight:.4f}")

# =====================================================================
# PHẦN 6: LƯU DỮ LIỆU THÀNH CÁC TẬP RIÊNG BIỆT
# =====================================================================
print("\nĐang lưu toàn bộ dữ liệu...")

# Lưu utils (LabelEncoder, TF-IDF, Class Weights)
with open("./preprocess/utils_and_weights.pkl", "wb") as f:
    pickle.dump({
        'label_encoder': le,
        'tfidf_vectorizer': tfidf,
        'class_weights': class_weights_idx
    }, f)

# Lưu Data cho SVM
with open("./preprocess/data_for_svm.pkl", "wb") as f:
    pickle.dump({
        'X_train': X_train_tfidf_smote,
        'y_train': y_train_smote,
        'X_test': X_test_tfidf,
        'y_test': y_test
    }, f)

# Lưu Data cho TextCNN & PhoBERT
df_train_dl = pd.DataFrame({'text': X_train_text, 'label': y_train})
df_test_dl = pd.DataFrame({'text': X_test_text, 'label': y_test})

df_train_dl.to_csv("./preprocess/train_textcnn_phobert.csv", index=False, encoding='utf-8-sig')
df_test_dl.to_csv("./preprocess/test_textcnn_phobert.csv", index=False, encoding='utf-8-sig')

print("\n✅ Đã hoàn tất! Dữ liệu đã chuẩn bị xong với mapping chính xác.")