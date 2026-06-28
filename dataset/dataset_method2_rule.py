import pandas as pd

# 1. Đọc dữ liệu
df2 = pd.read_csv('./dataset/dataset.csv')
if 'Unnamed: 3' in df2.columns:
    df2 = df2.drop(columns=['Unnamed: 3'])

df2['comment'] = df2['comment'].fillna('').astype(str)

# 2. Định nghĩa từ khóa mạnh
strong_pos = ['tuyệt vời', 'xuất sắc', 'đỉnh', 'hoàn hảo', 'dã man', 'xịn', '10 điểm', 'ưng ý', 'rất đẹp', 'rất tốt', 'rất nhanh', 'rất ưng', 'không thể chê']
strong_neg = ['tồi tệ', 'rách', 'lừa đảo', 'vứt', 'đừng mua', 'phí tiền', 'chán', 'tệ', 'hỏng', 'lởm', 'thất vọng', 'kém', 'rất xấu', 'quá đáng']

# 3. Hàm phân loại dựa trên luật
def rule_based_labeling(row):
    text = str(row['comment']).lower()
    old_label = row['label']
    
    if old_label == 'POS':
        if any(word in text for word in strong_pos): return 'Tuyệt vời'
        return 'Tốt'
    elif old_label == 'NEG':
        if any(word in text for word in strong_neg): return 'Rất tệ'
        return 'Tệ'
    else:
        return 'Bình thường'

# 4. Áp dụng hàm và lọc cột
df2['label_5_classes'] = df2.apply(rule_based_labeling, axis=1)
df2_final = df2[['comment', 'label_5_classes']]

# 5. Lưu thành dataset riêng
df2_final.to_csv('./dataset/dataset_method2_rule.csv', index=False, encoding='utf-8-sig')
print("Đã lưu: dataset_method2_rule.csv")