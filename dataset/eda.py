import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# 1. Đọc dữ liệu
df = pd.read_csv("./dataset/dataset.csv")

# Xem thông tin cơ bản
print("--- THÔNG TIN DỮ LIỆU ---")
print(df.info())

# 2. Tiền xử lý cơ bản
# Tập dữ liệu của bạn có một cột thừa 'Unnamed: 3' do lỗi parse CSV (dư dấu phẩy), ta sẽ loại bỏ nó
if 'Unnamed: 3' in df.columns:
    df = df.drop(columns=['Unnamed: 3'])

# Kiểm tra dữ liệu bị thiếu (Missing values)
print("\n--- DỮ LIỆU THIẾU ---")
print(df.isnull().sum())

# Xóa các dòng có comment bị thiếu (nếu có)
df = df.dropna(subset=['comment'])

# Kiểm tra và xóa dữ liệu trùng lặp
print("\nSố dòng trùng lặp:", df.duplicated().sum())
df = df.drop_duplicates()

# 3. Phân tích phân phối Nhãn (Label)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='label', palette='viridis', order=df['label'].value_counts().index)
plt.title('Phân phối các nhãn (Label Distribution)')
plt.xlabel('Nhãn')
plt.ylabel('Số lượng')
plt.show()

# 4. Phân tích phân phối Điểm đánh giá (Rate)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='rate', palette='magma')
plt.title('Phân phối điểm đánh giá (Rate Distribution)')
plt.xlabel('Điểm (1-5)')
plt.ylabel('Số lượng')
plt.show()

# 5. Phân tích mối quan hệ giữa Label và Rate
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='rate', hue='label', palette='Set2')
plt.title('Phân phối Nhãn theo Điểm đánh giá')
plt.xlabel('Điểm (1-5)')
plt.ylabel('Số lượng')
plt.legend(title='Nhãn')
plt.show()

# 6. Phân tích chiều dài văn bản (Text Length)
# Tính số ký tự (Character count)
df['char_count'] = df['comment'].apply(lambda x: len(str(x)))

# Tính số từ (Word count)
df['word_count'] = df['comment'].apply(lambda x: len(str(x).split()))

# Vẽ biểu đồ phân phối chiều dài theo số từ
plt.figure(figsize=(10, 6))
sns.histplot(df['word_count'], bins=50, kde=True, color='blue')
plt.title('Phân phối số từ trong bình luận')
plt.xlabel('Số từ')
plt.ylabel('Tần suất')
plt.xlim(0, 100) # Giới hạn hiển thị nếu có các bình luận quá dài gây nhiễu biểu đồ
plt.show()

# So sánh chiều dài văn bản (số từ) giữa các Nhãn
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='label', y='word_count', palette='pastel')
plt.title('Phân phối số từ theo từng Nhãn')
plt.xlabel('Nhãn')
plt.ylabel('Số từ')
plt.ylim(0, 100) 
plt.show()

# 7. Tạo WordCloud để xem các từ xuất hiện phổ biến nhất
text = " ".join(comment for comment in df['comment'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud - Các từ phổ biến trong toàn bộ bình luận', fontsize=20)
plt.show()

# (Tùy chọn) In ra một số thống kê mô tả cho chiều dài văn bản
print("\n--- THỐNG KÊ CHIỀU DÀI VĂN BẢN (SỐ TỪ) ---")
print(df.groupby('label')['word_count'].describe())