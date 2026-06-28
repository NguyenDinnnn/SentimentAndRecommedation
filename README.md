# Applying Machine Learning to Sentiment Analysis & Recommendation System

## 📌 Tính năng chính
* **Thu thập dữ liệu (Crawling)**: Tự động thu thập dữ liệu sản phẩm và các lượt đánh giá từ Tiki thông qua thư mục `crawl_tiki`.
* **Tiền xử lý (Preprocessing)**: Làm sạch dữ liệu, chuẩn hóa văn bản và phân tách từ tiếng Việt (word segmentation) để chuẩn bị cho quá trình huấn luyện.
* **Phân tích cảm xúc (Sentiment Analysis)**:
  * Triển khai đa dạng các kiến trúc: **SVM** (Machine Learning truyền thống), **TextCNN** với PyTorch (Deep Learning), và **PhoBERT** qua HuggingFace Transformers.
  * Tích hợp sẵn trọng số mô hình đã được huấn luyện (`textcnn_model.pth`).
* **Hệ thống gợi ý (Recommendation System)**: Module gợi ý sản phẩm phù hợp cho người dùng (`recommendation.py`).
* **Giao diện & API**: Cung cấp Web app hoàn chỉnh với frontend HTML/CSS và backend Python API (`app_api.py`).
* **Cơ sở dữ liệu**: Quản lý thông tin và lượng tồn kho sản phẩm bằng SQLite.

## Hướng dẫn cài đặt và sử dụng
* ** Bước 1: Clone repository
* git clone [(https://github.com/NguyenDinnnn/SentimentAndRecommedation)]
* cd SentimentAndRecommedation
* ** Bước 2: Cài đặt thư viện (Khuyến nghị sử dụng môi trường ảo (virtual environment). Chạy lệnh sau để cài đặt các thư viện cần thiết)
* python3 -m venv .venv
* pip install -r requirements.txt
* ** Bước 3: Khởi tạo CSDL (Khởi tạo database SQLite và nạp dữ liệu tồn kho ban đầu)
* python init_db.py
* python add_stock.py
* ** Bước 4: Khởi chạy ứng dụng Web / API
* python app_api.py
* ** Bước 5: Nếu không muốn sử dụng mà chỉ muốn dùng thử, hãy truy cập vào link: https://nguyndinnnn.id.vn/#home
