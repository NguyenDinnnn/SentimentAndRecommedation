import pandas as pd

# 1. Đọc dữ liệu
df1 = pd.read_csv('./dataset/dataset.csv')
if 'Unnamed: 3' in df1.columns:
    df1 = df1.drop(columns=['Unnamed: 3'])

# 2. Xử lý missing values
df1['comment'] = df1['comment'].fillna('').astype(str)

# 3. Khai báo từ điển ánh xạ từ cột rate
mapping_rate = {
    5: 'Tuyệt vời',
    4: 'Tốt',
    3: 'Bình thường',
    2: 'Tệ',
    1: 'Rất tệ'
}

# 4. Tạo cột nhãn mới và lọc các cột cần thiết
df1['label_5_classes'] = df1['rate'].map(mapping_rate)
df1_final = df1[['comment', 'label_5_classes']]

# 5. Lưu thành dataset riêng
df1_final.to_csv('dataset_method1_rate.csv', index=False, encoding='utf-8-sig')
print("Đã lưu: dataset_method1_rate.csv")