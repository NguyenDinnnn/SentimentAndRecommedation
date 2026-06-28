import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session, joinedload

def get_similar_products(db: Session, product_id: int, top_n: int = 4):
    from app_api import Product, ProductVariant
    """
    Thuật toán TF-IDF: Tính toán bằng Pandas, Trả dữ liệu an toàn
    """
    products = db.query(Product).all()
    if not products:
        return []

    data = [{'id': p.id, 'content': f"{p.name} {p.description or ''}"} for p in products]
    df = pd.DataFrame(data)

    if product_id not in df['id'].values:
        return []
    
    idx = df.index[df['id'] == product_id].tolist()[0]
    tfidf = TfidfVectorizer(stop_words='english') 
    tfidf_matrix = tfidf.fit_transform(df['content'])
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    similar_ids = [df.iloc[i[0]]['id'] for i in sim_scores]
    
    # TRUY VẤN LẠI DATABASE VÀ CHỈ LẤY SẢN PHẨM CÓ RATING >= 4.5
    recs = []
    for s_id in similar_ids:
        # Thêm điều kiện Product.rating >= 4.5
        p = db.query(Product).filter(Product.id == s_id, Product.rating >= 4.5).first()
        if p:
            first_valid_variant = db.query(ProductVariant).filter(
                ProductVariant.product_id == p.id,
                ProductVariant.stock > 0
            ).first()
            
            recs.append({
                "id": p.id,
                "name": p.name,
                "price": p.price,
                "image_url": p.image_url,
                "variant_id": first_valid_variant.id if first_valid_variant else 0 
            })
            
    return recs