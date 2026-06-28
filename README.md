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
* git clone [https://github.com/NguyenDinnnn/SandR.git]
* cd SandR
* ** Bước 2: Cài đặt thư viện (Khuyến nghị sử dụng môi trường ảo (virtual environment). Chạy lệnh sau để cài đặt các thư viện cần thiết)
* python3 -m venv .venv
* pip install -r requirements.txt
* ** Bước 3: Khởi tạo CSDL (Khởi tạo database SQLite và nạp dữ liệu tồn kho ban đầu)
* python init_db.py
* python add_stock.py
* ** Bước 4: Khởi chạy ứng dụng Web / API
* python app_api.py
* Sau khi server chạy, bạn có thể mở file index.html bằng cách truy cập (http://localhost:5500/#home) trên trình duyệt để sử dụng giao diện hệ thống.
* **** Trong trường hợp chạy Bước 4 gặp lỗi, hãy tạo 1 môi trường ảo .venv sau đó truy cập vào môi trường ảo đó.
* **** ** Sau đó chạy Server bằng câu lệnh: uvicorn app_api:app --host 0.0.0.0 --port 8093 --reload

## 📂 Cấu trúc dự án

```text
SentimentAndRecommedation/
├── crawl_tiki/           # Scripts cào dữ liệu đánh giá và sản phẩm từ Tiki
├── dataset/              # Chứa dữ liệu thô và dữ liệu đã qua xử lý (Vietnamese Sentiment Dataset)
├── phobert/              # Code huấn luyện và cấu hình cho mô hình PhoBERT
├── preprocess/           # Các hàm làm sạch văn bản, chuẩn hóa tiếng Việt
├── svm/                  # Code huấn luyện mô hình SVM baseline
├── textcnn/              # Kiến trúc TextCNN bằng PyTorch
├── textcnn_model.pth     # Trọng số của mô hình TextCNN đã được huấn luyện
├── recommendation.py     # Logic cốt lõi của Hệ thống gợi ý
├── app_api.py            # File chạy server chính (API + Backend)
├── init_db.py            # Khởi tạo schema cho database
├── add_stock.py          # Thêm/cập nhật dữ liệu hàng hóa
├── shopAI.db             # Database SQLite quản lý sản phẩm
├── index.html            # Giao diện frontend
└── style.css             # Style cho giao diện web
