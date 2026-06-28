import requests
import pandas as pd
import time

# Tạo headers giả lập trình duyệt để không bị Tiki chặn
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*'
}

# 1. Hàm lấy đánh giá của 1 sản phẩm (Truyền vào Product ID của Tiki)
def get_tiki_reviews(product_id, max_pages=5):
    reviews_data = []
    
    for page in range(1, max_pages + 1):
        # Đây là API thực tế của Tiki dùng để tải comment
        url = f'https://tiki.vn/api/v2/reviews?product_id={product_id}&page={page}&limit=10'
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                
                # Nếu trang này không có comment nào thì dừng
                if not data.get('data'):
                    break
                    
                for item in data['data']:
                    reviews_data.append({
                        'product_id': product_id,
                        'review_id': item.get('id'),
                        'rating': item.get('rating'),
                        'content': item.get('content', ''), # Chứa text review
                        'customer_name': item.get('created_by', {}).get('name', 'Ẩn danh')
                    })
                time.sleep(1) # Nghỉ 1s tránh bị block IP
            else:
                break
        except Exception as e:
            print(f"Lỗi: {e}")
            break
            
    return reviews_data

# Ví dụ chạy thử: Lấy đánh giá của 1 mã sản phẩm (Thay ID này bằng ID sản phẩm bạn muốn)
# Lên Tiki, bấm vào 1 sản phẩm, ID nằm trên URL (vd: p74021317.html -> ID là 74021317)
SAMPLE_PRODUCT_ID = '74021317' 
all_reviews = get_tiki_reviews(SAMPLE_PRODUCT_ID, max_pages=10)

# Lưu thành DataFrame và xuất CSV
df_reviews = pd.DataFrame(all_reviews)
# Lọc bỏ các đánh giá chỉ chấm sao mà không viết chữ
df_reviews = df_reviews[df_reviews['content'].str.strip() != '']
df_reviews.to_csv('./crawl_tiki/tiki_products_reviews.csv', index=False, encoding='utf-8-sig')
print(f"Đã crawl thành công {len(df_reviews)} đánh giá có text.")