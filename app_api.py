import os
import torch
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Boolean, DateTime, or_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, joinedload
from sqlalchemy.sql.expression import func
from transformers import PhobertTokenizer
from dotenv import load_dotenv

# ================= 1. DATABASE SETUP =================
# Load environment variables from .env if present
load_dotenv()
engine = create_engine("sqlite:///shopAI.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

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
    id = Column(Integer, primary_key=True, index=True)
    antecedent = Column(String, index=True)
    consequent = Column(String)
    confidence = Column(Float)

Base.metadata.create_all(bind=engine)

def get_similar_products(db, product_id: int, top_n: int = 4):
    """Get product similarity using the external recommendation.py helper."""
    try:
        from recommendation import get_similar_products as recommendation_get_similar_products
    except ImportError:
        return []

    try:
        return recommendation_get_similar_products(db, product_id, top_n=top_n)
    except Exception:
        return []

# ================= 2. FASTAPI & PHOBERT =================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

print("Dang tai AI PhoBERT tu Hugging Face hub...")

try:
    from phobert.phobert import tokenizer, model
    print("Da load AI PhoBERT tu Hugging Face hub!")
except Exception as e:
    print("Chay Mock AI Mode (Khong tai duoc model tu Hugging Face):", e)
    tokenizer, model = None, None

label_map = {1: "Rất tệ", 2: "Tệ", 3: "Trung tính", 4: "Tốt", 5: "Tuyệt vời"} 


@app.get("/config.js")
def config_js(request: Request):
    """Serve a small JS snippet that sets window.APP_CONFIG.apiBaseUrl from env vars.
    This lets the frontend read API base URL at runtime without hardcoding.
    """
    # Prefer BASE_DOMAIN, then FRONTEND_API_URL, then use the current request origin
    domain = os.getenv("BASE_DOMAIN") or os.getenv("FRONTEND_API_URL")
    if not domain:
        domain = str(request.base_url).rstrip('/')
    api_base = domain.rstrip('/') + '/api'
    js = f"window.APP_CONFIG = {{ apiBaseUrl: '{api_base}' }};"
    return Response(content=js, media_type="application/javascript")

# ================= 3.0 HOMEPAGE ENDPOINTS =================
@app.get("/")
async def index():
    return FileResponse("templates/index.html")

@app.get("/favicon.svg")
async def favicon():
    return FileResponse("templates/favicon.svg", media_type="image/svg+xml")

@app.get("/style.css")
async def style():
    return FileResponse("templates/style.css", media_type="text/css")

# ================= 3. API ENDPOINTS =================

@app.get("/api/products")
async def get_products():
    db = SessionLocal()
    products = db.query(Product).options(joinedload(Product.shop)).order_by(func.random()).limit(30).all()
    db.close()
    return [{"id": p.id, "name": p.name, "price": p.price, "original_price": p.original_price, "discount_rate": p.discount_rate, "image_url": p.image_url, "sold": p.sold, "shop_name": p.shop.name} for p in products]

@app.get("/api/product/{product_id}")
async def get_product_detail(product_id: int):
    db = SessionLocal()
    try:
        p = db.query(Product).options(joinedload(Product.shop), joinedload(Product.variants)).filter(Product.id == product_id).first()
        if not p:
            raise HTTPException(status_code=404, detail="Không tìm thấy sản phẩm")

        # 1. Gợi ý Tương tự (TF-IDF)
        similar_recs = get_similar_products(db, product_id, top_n=4)
        
        # Hàm an toàn: Lấy ID phân loại còn hàng
        def get_safe_variant_id(product_obj):
            if product_obj.variants:
                for v in product_obj.variants:
                    if v.stock and v.stock > 0:
                        return v.id
            return 0 

        # 2. Gợi ý Mua kèm (Apriori)
        apriori_recs = []
        rules = db.query(MatchingRule).all()
        consequent_names = [r.consequent for r in rules if r.antecedent.lower() in p.name.lower()]
                
        if consequent_names:
            apriori_products = db.query(Product).options(joinedload(Product.variants)).filter(
                or_(*[Product.name.contains(name) for name in consequent_names])
            ).filter(Product.id != p.id).limit(4).all()
            
            apriori_recs = [{"id": r.id, "name": r.name, "price": r.price, "image_url": r.image_url, "variant_id": get_safe_variant_id(r)} for r in apriori_products]
        
        # Nếu Apriori trống, lấy ngẫu nhiên
        if not apriori_recs:
            random_products = db.query(Product).options(joinedload(Product.variants)).filter(Product.id != p.id).order_by(func.random()).limit(4).all()
            apriori_recs = [{"id": r.id, "name": r.name, "price": r.price, "image_url": r.image_url, "variant_id": get_safe_variant_id(r)} for r in random_products]

        return {
            "product": {
                "id": p.id, "name": p.name, "description": p.description, "price": p.price, "original_price": p.original_price, "discount_rate": p.discount_rate, "image_url": p.image_url, "rating": round(p.rating, 1) if p.rating else 0, "review_count": p.review_count or 0,
                "shop": {"id": p.shop.id, "name": p.shop.name} if p.shop else None
            },
            "variants": [{"id": v.id, "color": v.color, "size": v.size, "stock": v.stock} for v in p.variants] if p.variants else [],
            "recommendations_similar": similar_recs,
            "recommendations_apriori": apriori_recs 
        }
    except Exception as e:
        print(f"❌ Lỗi sập API lấy sản phẩm: {e}") # Báo lỗi ra Terminal để bạn biết ngay
        raise HTTPException(status_code=500, detail="Lỗi nội bộ hệ thống")
    finally:
        db.close()

class AddCartRequest(BaseModel):
    variant_id: int
    quantity: int = 1

@app.post("/api/cart")
async def add_to_cart(request: AddCartRequest):
    db = SessionLocal()
    existing_item = db.query(CartItem).filter(CartItem.variant_id == request.variant_id).first()
    if existing_item:
        existing_item.quantity += request.quantity
    else:
        db.add(CartItem(variant_id=request.variant_id, quantity=request.quantity))
    db.commit()
    db.close()
    return {"status": "success", "message": "Đã thêm vào giỏ hàng!"}

class UpdateCartRequest(BaseModel):
    quantity: int

@app.get("/api/cart")
async def get_cart():
    db = SessionLocal()
    # THAY ĐỔI TẠI ĐÂY: Thêm .order_by(CartItem.id.desc()) để sắp xếp sản phẩm mới lên đầu
    cart_items = db.query(CartItem).options(
        joinedload(CartItem.variant).joinedload(ProductVariant.product).joinedload(Product.shop)
    ).order_by(CartItem.id.desc()).all()
    
    result = []
    for item in cart_items:
        product = item.variant.product
        result.append({
            "cart_item_id": item.id,
            "variant_id": item.variant.id,
            "product_id": product.id,
            "color": item.variant.color,
            "size": item.variant.size,
            "quantity": item.quantity,
            "stock": item.variant.stock, 
            "product_name": product.name,
            "price": product.price,
            "image_url": product.image_url,
            "shop_name": product.shop.name
        })
    db.close()
    return result

@app.get("/api/review/{order_item_id}")
async def get_review(order_item_id: int):
    db = SessionLocal()
    review = db.query(Review).filter(Review.order_item_id == order_item_id).first()
    db.close()
    if not review:
        raise HTTPException(status_code=404, detail="Chưa có đánh giá")
    
    # Tính số ngày đã trôi qua
    delta = datetime.utcnow() - review.created_at
    days_left = max(0, 7 - delta.days)
    can_edit = delta.days <= 7
    
    return {
        "star_rating": review.star_rating,
        "review_text": review.review_text,
        "sentiment_label": review.sentiment_label,
        "can_edit": can_edit,
        "days_left": days_left
    }

@app.put("/api/cart/{cart_item_id}")
async def update_cart_item(cart_item_id: int, request: UpdateCartRequest):
    db = SessionLocal()
    item = db.query(CartItem).filter(CartItem.id == cart_item_id).first()
    if not item:
        db.close()
        raise HTTPException(status_code=404, detail="Không tìm thấy sản phẩm")
    
    item.quantity = request.quantity
    db.commit()
    db.close()
    return {"status": "success"}

@app.delete("/api/cart/{cart_item_id}")
async def delete_cart_item(cart_item_id: int):
    db = SessionLocal()
    item = db.query(CartItem).filter(CartItem.id == cart_item_id).first()
    if not item:
        db.close()
        raise HTTPException(status_code=404, detail="Không tìm thấy sản phẩm trong giỏ hàng")
    db.delete(item)
    db.commit()
    db.close()
    return {"status": "success", "message": "Sản phẩm đã được xóa khỏi giỏ hàng"}

@app.delete("/api/cart")
async def clear_cart():
    db = SessionLocal()
    db.query(CartItem).delete()
    db.commit()
    db.close()
    return {"status": "success", "message": "Đã xóa toàn bộ giỏ hàng"}

class CheckoutRequest(BaseModel):
    cart_item_ids: list[int]

@app.post("/api/checkout")
async def checkout(request: CheckoutRequest):
    db = SessionLocal()
    cart_items = db.query(CartItem).options(joinedload(CartItem.variant).joinedload(ProductVariant.product)).filter(CartItem.id.in_(request.cart_item_ids)).all()
    
    if not cart_items:
        db.close()
        return {"status": "error", "message": "Không có sản phẩm nào được chọn!"}
    
    new_order = Order(total_price=0, status="SUCCESS")
    db.add(new_order)
    db.commit() 
    
    total = 0
    for c_item in cart_items:
        price = c_item.variant.product.price
        total += price * c_item.quantity
        db.add(OrderItem(order_id=new_order.id, variant_id=c_item.variant_id, quantity=c_item.quantity, price=price, is_reviewed=False))
        db.delete(c_item)
        
    new_order.total_price = total
    db.commit()
    db.close()
    return {"status": "success", "message": "Thanh toán thành công!"}

@app.get("/api/history")
async def get_history():
    db = SessionLocal()
    # Nếu bạn cũng muốn đơn hàng mới nhất lên đầu (như giỏ hàng), nó đã có sẵn Order.created_at.desc()
    orders = db.query(Order).options(joinedload(Order.items).joinedload(OrderItem.variant).joinedload(ProductVariant.product).joinedload(Product.shop)).order_by(Order.created_at.desc()).all()
    
    result = []
    for order in orders:
        items = []
        for item in order.items:
            product = item.variant.product
            items.append({
                "order_item_id": item.id,
                "product_id": product.id,
                "product_name": product.name,
                "variant_info": f"{item.variant.color}, {item.variant.size}",
                "price": item.price,
                "quantity": item.quantity,
                "image_url": product.image_url,
                "shop_name": product.shop.name,
                "is_reviewed": item.is_reviewed
            })
        
        result.append({
            "order_id": order.id,
            "total_price": order.total_price,
            "created_at": order.created_at,
            "status": order.status,
            "items": items
        })
        
    db.close()
    return result

class ReviewRequest(BaseModel):
    order_item_id: int
    review_text: Optional[str] = "" # Nếu không gửi text, mặc định là chuỗi rỗng
    star_rating: Optional[int] = 0  # Nếu không gửi sao, mặc định là 0

from datetime import datetime

@app.post("/api/evaluate")
async def evaluate_review(request: ReviewRequest):
    db = SessionLocal()
    
    text = request.review_text.strip() if request.review_text else ""
    star_score = request.star_rating
    
    # 1. Bắt lỗi bắt buộc (Thiếu 1 trong 2 là báo lỗi)
    if text == "" or star_score == 0:
        db.close()
        return {"status": "error", "message": "Bắt buộc phải tick sao và nhập bình luận!"}

    # 2. Bắt lỗi văn bản rác (Spam filter cơ bản)
    words = text.split()
    if len(words) < 3 or len(text) < 10:
        db.close()
        return {"status": "error", "message": "Nội dung đánh giá không hợp lệ. Vui lòng nhập có ý nghĩa hơn."}

    # 3. Chấm điểm bằng PhoBERT hoặc Từ khóa
    phobert_score = 0
    if model and tokenizer:
        inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=120)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        star_feature = torch.tensor([[float(star_score) / 5.0]], dtype=torch.float).to(device)
        with torch.no_grad():
            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], star_feature=star_feature)
            predicted_class_id = torch.argmax(logits, dim=1).item()
            phobert_to_score = {1: 1, 4: 2, 0: 3, 3: 4, 2: 5}
            phobert_score = phobert_to_score.get(predicted_class_id, 3)
    else:
        text_lower = text.lower()
        if any(word in text_lower for word in ["tốt", "đẹp", "thích", "nhanh", "ok", "tuyệt"]): phobert_score = 5
        elif any(word in text_lower for word in ["tệ", "chậm", "chật", "xấu", "lỗi", "không"]): phobert_score = 1
        else: phobert_score = 3

    # 4. Tính điểm lai ghép (Đã rút gọn vì chắc chắn có cả 2)
    final_score = round((phobert_score + star_score) / 2)
        
    score_to_sentiment = {1: "Rất tệ", 2: "Tệ", 3: "Bình thường", 4: "Tốt", 5: "Tuyệt vời"}
    sentiment = score_to_sentiment.get(final_score, "Bình thường")
    detailed_sentiment = f"{sentiment} (Điểm tổng: {final_score}/5)"
    
    # =========================================================
    # 4. LƯU DATABASE & CẬP NHẬT TRẠNG THÁI (ĐOẠN FIX LỖI)
    # =========================================================
    review = db.query(Review).filter(Review.order_item_id == request.order_item_id).first()
    action_text = "đã phân tích"
    
    if review:
        # Xử lý cập nhật đánh giá (nếu đã có)
        delta = datetime.utcnow() - review.created_at
        if delta.days > 7:
            db.close()
            return {"status": "error", "message": "Đã quá hạn 7 ngày, bạn không thể chỉnh sửa đánh giá này nữa."}
        
        review.star_rating = star_score
        review.review_text = text
        review.sentiment_label = detailed_sentiment
        action_text = "đã cập nhật"
        db.commit() # Lưu vào DB
    else:
        # Xử lý đánh giá mới lần đầu
        new_review = Review(
            order_item_id=request.order_item_id, 
            star_rating=star_score, 
            review_text=text, 
            sentiment_label=detailed_sentiment
        )
        db.add(new_review)
        
        # ĐỔI TRẠNG THÁI SẢN PHẨM THÀNH "ĐÃ ĐÁNH GIÁ"
        order_item_update = db.query(OrderItem).filter(OrderItem.id == request.order_item_id).first()
        if order_item_update:
            order_item_update.is_reviewed = True
        
        db.commit() # Lưu vào DB

    # =========================================================
    # 5. MỞ RỘNG LOGIC GỢI Ý CHO 5 CẤP ĐỘ
    # =========================================================
    order_item = db.query(OrderItem).options(joinedload(OrderItem.variant)).filter(OrderItem.id == request.order_item_id).first()
    product_id_to_match = order_item.variant.product_id if order_item else 1
    
    recs = []
    message = ""
    
    if sentiment == "Tuyệt vời":
        recs_db = db.query(Product).order_by(func.random()).limit(3).all()
        recs = [{"id": r.id, "name": r.name, "price": r.price, "image_url": r.image_url} for r in recs_db]
        message = f"Thật tuyệt! AI {action_text} đánh giá là: {detailed_sentiment}. Mọi người thường mua thêm các sản phẩm này:"
    elif sentiment == "Tốt":
        recs_db = db.query(Product).order_by(func.random()).limit(3).all()
        recs = [{"id": r.id, "name": r.name, "price": r.price, "image_url": r.image_url} for r in recs_db]
        message = f"Cảm ơn bạn! AI {action_text} đánh giá là: {detailed_sentiment}. Dưới đây là các sản phẩm bạn có thể thích:"
    elif sentiment == "Bình thường":
        recs_db = db.query(Product).order_by(func.random()).limit(3).all()
        recs = [{"id": r.id, "name": r.name, "price": r.price, "image_url": r.image_url} for r in recs_db]
        message = f"AI {action_text} đánh giá của bạn là: {detailed_sentiment}. Cảm ơn bạn đã đóng góp ý kiến."
    def get_safe_variant_id(product_obj):
        if hasattr(product_obj, 'variants') and product_obj.variants:
            for v in product_obj.variants:
                if v.stock and v.stock > 0:
                    return v.id
        return 0

    def build_fallback_products(products):
        return [
            {
                "id": r.id,
                "name": r.name,
                "price": r.price,
                "image_url": r.image_url,
                "variant_id": get_safe_variant_id(r)
            }
            for r in products
        ]

    if sentiment == "Tệ":
        recs = get_similar_products(db, product_id_to_match, top_n=3)
        if not recs:
            # Bổ sung điều kiện Product.rating >= 4.5 để lấy sản phẩm chất lượng cao thay thế
            fallback_products = db.query(Product).options(joinedload(Product.variants))\
                                  .filter(Product.id != product_id_to_match, Product.rating >= 4.5)\
                                  .order_by(func.random()).limit(3).all()
            recs = build_fallback_products(fallback_products)
        message = f"Rất tiếc vì trải nghiệm {detailed_sentiment}. Mời bạn tham khảo các sản phẩm thay thế có chất lượng phù hợp hơn:"
        
    elif sentiment == "Rất tệ":
        recs = get_similar_products(db, product_id_to_match, top_n=3)
        if not recs:
            # Bổ sung điều kiện Product.rating >= 4.5 để lấy sản phẩm chất lượng cao thay thế
            fallback_products = db.query(Product).options(joinedload(Product.variants))\
                                  .filter(Product.id != product_id_to_match, Product.rating >= 4.5)\
                                  .order_by(func.random()).limit(3).all()
            recs = build_fallback_products(fallback_products)
        message = f"Hệ thống ghi nhận lỗi nghiêm trọng: {detailed_sentiment}. Quản lý CSKH sẽ liên hệ với bạn ngay lập tức. Xin gửi bạn các lựa chọn thay thế tốt nhất:"
        
    db.close()
    
    return {
        "status": "success",
        "ai_analysis": {"sentiment": detailed_sentiment, "message": message},
        "recommendations": recs
    }