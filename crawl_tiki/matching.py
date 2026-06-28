import pandas as pd
import random
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sqlalchemy import create_engine

print("================ HỆ THỐNG GỢI Ý MUA SẮM (APRIORI) ================\n")

# ---------------------------------------------------------
# BƯỚC 1: LẤY DỮ LIỆU THẬT & GIẢ LẬP GIỎ HÀNG THÔNG MINH
# ---------------------------------------------------------
print("1. Đang kết nối Database lấy danh sách sản phẩm...")
try:
    engine = create_engine("sqlite:///shopAI.db")
    df_products = pd.read_sql("SELECT id, name, shop_id FROM products", engine)
except Exception as e:
    print(f"❌ Lỗi kết nối Database: {e}")
    exit()

if df_products.empty:
    print("❌ Database trống. Hãy chạy init_db.py trước để có dữ liệu sản phẩm!")
    exit()

print(f"✅ Đã tải {len(df_products)} sản phẩm. Bắt đầu tạo 10.000 giỏ hàng...")
transactions = []

# Nhóm sản phẩm theo shop_id để mô phỏng hành vi: "Khách thường mua nhiều món cùng 1 shop"
shop_groups = df_products.groupby('shop_id')['name'].apply(list).to_dict()
valid_shops = {k: v for k, v in shop_groups.items() if len(v) >= 2}

for _ in range(10000):
    basket = []
    
    # 85% trường hợp khách sẽ chọn mua nhiều đồ trong cùng 1 shop
    if valid_shops and random.random() < 0.85: 
        shop_id = random.choice(list(valid_shops.keys()))
        shop_products = valid_shops[shop_id]
        
        # Mua từ 2 đến tối đa 4 món của shop đó
        num_items = random.randint(2, min(4, len(shop_products)))
        basket.extend(random.sample(shop_products, num_items))
    else:
        # 15% trường hợp khách mua ngẫu nhiên chéo shop
        num_items = random.randint(1, 3)
        basket.extend(df_products['name'].sample(num_items).tolist())
            
    if len(basket) > 1: # Thuật toán chỉ học trên các hóa đơn mua >= 2 món
        transactions.append(basket)

print(f"✅ Đã tạo xong {len(transactions)} hóa đơn hợp lệ.")
print("Ví dụ hóa đơn số 1:\n -", "\n - ".join(transactions[0]))

# ---------------------------------------------------------
# BƯỚC 2: CHUYỂN ĐỔI DỮ LIỆU ĐỂ THUẬT TOÁN ĐỌC ĐƯỢC
# ---------------------------------------------------------
print("\n2. Đang One-Hot Encoding dữ liệu...")
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_basket = pd.DataFrame(te_ary, columns=te.columns_)

# ---------------------------------------------------------
# BƯỚC 3: CHẠY THUẬT TOÁN APRIORI ĐỂ TÌM LUẬT GỢI Ý
# ---------------------------------------------------------
print("\n3. Đang huấn luyện thuật toán Apriori...")
# Hạ min_support xuống một chút vì số lượng sản phẩm thật đa dạng hơn
frequent_itemsets = apriori(df_basket, min_support=0.01, use_colnames=True)

# min_threshold = 0.4: Lấy luật có độ tin cậy >= 40%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)

if rules.empty:
    print("⚠️ Không tìm thấy luật gợi ý nào đạt chuẩn! Thử tăng số lượng hóa đơn hoặc giảm min_support.")
    exit()

rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

# Xử lý chuỗi kết quả
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
final_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

print("\n✅ HOÀN TẤT! TOP 5 LUẬT GỢI Ý MUA KÈM:")
print(final_rules.head(5).to_string(index=False))

import os
os.makedirs('./crawl_tiki', exist_ok=True)
final_rules.to_csv('./crawl_tiki/matching_rules.csv', index=False, encoding='utf-8-sig')
print("\n✅ Đã lưu bộ luật mới vào file 'matching_rules.csv'. Bạn có thể chạy lại init_db.py để nạp vào DB!")