import os
import random
import time
import requests
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

if os.path.exists("shopAI.db"):
    os.remove("shopAI.db")
    print("🗑️ Đã xóa Database cũ...")

engine = create_engine("sqlite:///shopAI.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ================= 1. SCHEMA =================
class Shop(Base):
    __tablename__ = "shops"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    logo_url = Column(String)

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    shop_id = Column(Integer, ForeignKey("shops.id"))
    name = Column(String, index=True)
    original_price = Column(Integer)
    price = Column(Integer)
    discount_rate = Column(Integer)
    sold = Column(Integer)
    rating = Column(Float)
    review_count = Column(Integer)
    image_url = Column(String)
    description = Column(String)
    
    shop = relationship("Shop")
    variants = relationship("ProductVariant", back_populates="product", cascade="all, delete-orphan")

class ProductVariant(Base):
    __tablename__ = "product_variants"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    color = Column(String)
    size = Column(String)
    stock = Column(Integer)
    
    product = relationship("Product", back_populates="variants")

class CartItem(Base):
    __tablename__ = "cart_items"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    variant_id = Column(Integer, ForeignKey("product_variants.id"))
    quantity = Column(Integer)
    
    variant = relationship("ProductVariant")

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    total_price = Column(Integer)
    status = Column(String, default="SUCCESS")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    items = relationship("OrderItem", back_populates="order")

class OrderItem(Base):
    __tablename__ = "order_items"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    variant_id = Column(Integer, ForeignKey("product_variants.id"))
    quantity = Column(Integer)
    price = Column(Integer)
    is_reviewed = Column(Boolean, default=False)
    
    order = relationship("Order", back_populates="items")
    variant = relationship("ProductVariant")

class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    order_item_id = Column(Integer, ForeignKey("order_items.id"))
    star_rating = Column(Integer)
    review_text = Column(String)
    sentiment_label = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class MatchingRule(Base):
    __tablename__ = "matching_rules"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    antecedent = Column(String, index=True)
    consequent = Column(String)
    confidence = Column(Float)

Base.metadata.create_all(bind=engine)
db = SessionLocal()

# ================= 2. CRAWL DATA TỪ NHIỀU DANH MỤC =================
print("⏳ Đang cào dữ liệu từ Tiki...")
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# 8322: Thời trang, 1520: Làm đẹp, 1815: Thiết bị số
categories = ["8322", "1520", "1815"] 
shop_dict = {}
total_products = 0

for cat in categories:
    api_url = f"https://tiki.vn/api/personalish/v1/blocks/listings?limit=40&category={cat}"
    data = []
    for attempt in range(1, 4):
        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json().get('data', [])
            break
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Lỗi kết nối danh mục {cat}, lần {attempt}/3: {e}")
            if attempt == 3:
                print(f"❌ Bỏ qua danh mục {cat} sau 3 lần thử.")
            else:
                time.sleep(2)
    
    if not data:
        continue
    
    # Thiết lập thuộc tính theo danh mục
    if cat == "8322": # Thời trang
        colors_mock = ["Trắng", "Đen", "Be", "Đỏ"]
        sizes_mock = ["S", "M", "L", "XL"]
    elif cat == "1520": # Làm đẹp
        colors_mock = ["Mặc định"]
        sizes_mock = ["50ml", "100ml", "Da khô", "Da dầu"]
    else: # Thiết bị số
        colors_mock = ["Titan", "Đen Nhám", "Bạc"]
        sizes_mock = ["64GB", "128GB", "256GB"]
    
    try:
        for item in data:
            seller = item.get('seller_product_attributes') or {}
            seller_name = seller.get('seller_name', 'Tiki Trading')
            seller_id = seller.get('seller_id', random.randint(1000, 9999))
            
            if seller_id not in shop_dict:
                new_shop = Shop(
                    id=seller_id, 
                    name=seller_name, 
                    logo_url=seller.get('logo', 'https://salt.tikicdn.com/ts/seller/21/c4/88/f26ea4ba6dbb200b2ce28bba33d0144c.jpg')
                )
                db.add(new_shop)
                shop_dict[seller_id] = True
                db.commit()
                
            product_id = item.get('id')
            img_url = item.get('thumbnail_url') or f"https://picsum.photos/seed/{product_id}/400/400"
            sold_data = item.get('quantity_sold') or {}
            sold_value = sold_data.get('value', random.randint(10, 500))

            new_product = Product(
                id=product_id,
                shop_id=seller_id,
                name=item.get('name'),
                original_price=item.get('original_price', item.get('price', 0)),
                price=item.get('price', 0),
                discount_rate=item.get('discount_rate', 0),
                sold=sold_value,
                rating=item.get('rating_average') or random.uniform(3.5, 5.0),
                review_count=item.get('review_count') or random.randint(5, 100),
                image_url=img_url,
                description=f"Sản phẩm {item.get('name')} chính hãng."
            )
            db.merge(new_product)
            total_products += 1
            
            # Khởi tạo variant linh hoạt
            num_variants = random.randint(1, 3)
            for _ in range(num_variants):
                new_variant = ProductVariant(
                    product_id=product_id,
                    color=random.choice(colors_mock),
                    size=random.choice(sizes_mock),
                    stock=random.randint(10, 100)
                )
                db.add(new_variant)

        db.commit()
    except Exception as e:
        print(f"❌ Lỗi crawl danh mục {cat}: {e}")

print(f"✅ Đã lưu thành công {total_products} sản phẩm vào Database!")

# ================= 3. TẠO DATA ĐƠN HÀNG VÀ NẠP REVIEWS TỪ CSV =================
print("⏳ Đang thiết lập lịch sử mua hàng và Reviews từ CSV...")
try:
    # Đọc file CSV chứa dữ liệu cào thật
    df_reviews = pd.read_csv('./crawl_tiki/tiki_products_reviews.csv')
    
    # Lấy toàn bộ biến thể hiện có
    all_variants = db.query(ProductVariant).all()
    if not all_variants:
        raise ValueError("Không có variant nào trong CSDL để tạo đơn hàng.")

    # Lặp qua từng review trong CSV để tạo đơn hàng tương ứng
    saved_reviews = 0
    for _, row in df_reviews.iterrows():
        review_product_id = row.get('product_id')
        
        # Tìm variant khớp với product_id của review, nếu không có thì random
        matched_variants = [v for v in all_variants if v.product_id == review_product_id]
        if matched_variants:
            chosen_variant = random.choice(matched_variants)
        else:
            chosen_variant = random.choice(all_variants)

        # Tạo đơn hàng
        new_order = Order(total_price=chosen_variant.product.price)
        db.add(new_order)
        db.commit()

        # Tạo Item trong đơn hàng
        order_item = OrderItem(
            order_id=new_order.id, 
            variant_id=chosen_variant.id, 
            quantity=1, 
            price=chosen_variant.product.price, 
            is_reviewed=True
        )
        db.add(order_item)
        db.commit()

        # Áp dụng logic phân loại sắc thái cơ bản (có thể tích hợp model ML tại đây)
        rating = int(row.get('rating', 5))
        if rating >= 4:
            sentiment = "Tích cực"
        elif rating == 3:
            sentiment = "Trung tính"
        else:
            sentiment = "Tiêu cực"

        # Thêm review thật
        new_review = Review(
            order_item_id=order_item.id,
            star_rating=rating,
            review_text=str(row.get('content', '')),
            sentiment_label=sentiment
        )
        db.add(new_review)
        saved_reviews += 1

    db.commit()
    print(f"✅ Đã tạo đơn hàng và nạp thành công {saved_reviews} đánh giá thật từ Tiki!")

except FileNotFoundError:
    print("⚠️ Không tìm thấy './crawl_tiki/tiki_products_reviews.csv'. Vui lòng chạy file crawl review trước.")
except Exception as e:
    print(f"❌ Lỗi khi nạp reviews: {e}")

# ================= 4. NẠP LUẬT APRIORI TỪ FILE =================
print("⏳ Đang nạp luật Apriori từ matching_rules.csv...")
try:
    df_rules = pd.read_csv('./crawl_tiki/matching_rules.csv')
    saved_rules = 0
    for _, row in df_rules.iterrows():
        antecedents_str = str(row['antecedents']).replace("frozenset({'", "").replace("'})", "").strip()
        consequents_str = str(row['consequents']).replace("frozenset({'", "").replace("'})", "").strip()
        
        new_rule = MatchingRule(
            antecedent=antecedents_str,
            consequent=consequents_str,
            confidence=float(row['confidence'])
        )
        db.add(new_rule)
        saved_rules += 1
        
    db.commit()
    print(f"✅ Đã nạp thành công {saved_rules} luật bán chéo (Cross-sell)!")
except FileNotFoundError:
    print("⚠️ Không tìm thấy 'matching_rules.csv'. Hãy chạy file matching.py!")

db.close()
print("🎉 HOÀN TẤT SETUP DATABASE!")