import pandas as pd
import re
from tqdm import tqdm

# 2. Đọc dữ liệu
df3 = pd.read_csv('./dataset/dataset.csv')
if 'Unnamed: 3' in df3.columns:
    df3 = df3.drop(columns=['Unnamed: 3'])

df3['comment'] = df3['comment'].fillna('').astype(str)

# 100 dòng đầu tiên
df3 = df3.head(100).copy()

# 3. Từ điển keywords cho mỗi cảm xúc
sentiment_keywords = {
    "Tuyệt vời": [
        "tuyệt vời", "siêu yêu", "đẹp quá", "quá đẹp", "rất thích", "yêu thích",
        "tuyệt", "superb", "excellent", "amazing", "perfect", "awesome", "loveit",
        "best", "tốt tuyệt", "cực tốt", "cực đẹp", "siêu", "siêu chi là"
    ],
    "Tốt": [
        "tốt", "ổn", "okz", "oki", "ok", "nice", "good", "giỏi", "hài lòng",
        "đáng tiền", "rất ổn", "chất lượng tốt", "vừa", "khỏi", "đủ", "được",
        "chắc chắn", "gọn gàng", "đẹp", "chất", "mềm", "mịn", "dày", "nhanh",
        "ngon", "re", "rẻ", "tốt tính", "nhiệt tình", "tâm"
    ],
    "Bình thường": [
        "bình thường", "trung bình", "tạm", "tạm ổn", "tạm được", "như vậy",
        "khá", "vừa vặn", "chưa thật sự", "hơi", "còn", "hoàn toàn", "cũng đc",
        "tạm thể", "bỏ qua", "chịu", "được rồi", "chập chùng"
    ],
    "Tệ": [
        "tệ", "dở", "xấu", "thất vọng", "khó", "lỏng", "xù", "thô", "cứng",
        "mỏng", "hượng", "tệ hại", "lừa", "lừa đảo", "không", "k", "khó mặc",
        "ko được", "không được", "hại", "hỏng", "sệt", "nó vứt", "bích", "khô"
    ],
    "Rất tệ": [
        "rất tệ", "cực tệ", "tệ hơn", "kinh hoạc", "kinh khủng", "ghê gớm",
        "tệ nhất", "tệ hại nhất", "lừa đảo", "fake", "giả", "hàng fake",
        "nhem", "bẩn", "sêu bẩn", "nát", "bục", "rách", "nứt", "hỏng hoàn toàn"
    ]
}

# 4. Hàm phân loại dựa trên keywords
def classify_sentiment(text):
    text_lower = text.lower()
    text_no_accent = text_lower.replace('á', 'a').replace('à', 'a').replace('ả', 'a').replace('ã', 'a').replace('ạ', 'a')
    text_no_accent = text_no_accent.replace('é', 'e').replace('è', 'e').replace('ẻ', 'e').replace('ẽ', 'e').replace('ẹ', 'e')
    text_no_accent = text_no_accent.replace('í', 'i').replace('ì', 'i').replace('ỉ', 'i').replace('ĩ', 'i').replace('ị', 'i')
    text_no_accent = text_no_accent.replace('ó', 'o').replace('ò', 'o').replace('ỏ', 'o').replace('õ', 'o').replace('ọ', 'o')
    text_no_accent = text_no_accent.replace('ú', 'u').replace('ù', 'u').replace('ủ', 'u').replace('ũ', 'u').replace('ụ', 'u')
    text_no_accent = text_no_accent.replace('ý', 'y').replace('ỳ', 'y').replace('ỷ', 'y').replace('ỹ', 'y').replace('ỵ', 'y')
    
    scores = {}
    for sentiment, keywords in sentiment_keywords.items():
        count = 0
        for keyword in keywords:
            if keyword in text_lower or keyword in text_no_accent:
                count += 1
        scores[sentiment] = count
    
    # Nếu có match, trả về sentiment với score cao nhất
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    else:
        return "Bình thường"  # Default label

# 5. Áp dụng phân loại
print("Đang phân loại cảm xúc dựa trên keywords...")
tqdm.pandas()
df3['label_5_classes'] = df3['comment'].progress_apply(classify_sentiment)

# 6. Lưu thành dataset
df3_final = df3[['comment', 'label_5_classes']]
df3_final.to_csv('./dataset/dataset_method3_llm.csv', index=False, encoding='utf-8-sig')
print("Đã lưu: dataset_method3_llm.csv")

# Thống kê
print("\nThống kê kết quả phân loại:")
print(df3_final['label_5_classes'].value_counts())
print("Đã lưu: dataset_method3_llm.csv")