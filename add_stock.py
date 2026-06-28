from app_api import SessionLocal, Product, ProductVariant

def fill_empty_stock():
    db = SessionLocal()
    print("📦 Đang kiểm tra kho hàng...")
    
    products = db.query(Product).all()
    updated_count = 0
    
    for p in products:
        # Nếu sản phẩm chưa có bất kỳ phân loại (variant) nào
        if not p.variants:
            # Tạo một phân loại mặc định với số lượng tồn kho là 100
            default_variant = ProductVariant(
                product_id=p.id,
                color="Mặc định",
                size="Freesize",
                stock=100
            )
            db.add(default_variant)
            updated_count += 1
            
    # Lưu toàn bộ thay đổi vào CSDL
    db.commit()
    db.close()
    
    if updated_count > 0:
        print(f"✅ Thành công! Đã bơm hàng (Tồn kho: 100) cho {updated_count} sản phẩm bị thiếu.")
    else:
        print("👌 Kho hàng của bạn đã đầy đủ, không có sản phẩm nào bị trống.")

if __name__ == "__main__":
    fill_empty_stock()