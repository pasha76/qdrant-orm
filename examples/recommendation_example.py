"""
Example usage of recommendation functionality in Qdrant ORM
"""
import numpy as np
import sys
sys.path.append("/Users/tolgagunduz/Documents/projects/blushyv3/orm/")

from qdrant_orm import (
    Base, Field, VectorField, ArrayField,
    QdrantEngine, QdrantSession,
    String, Integer, Float, Boolean, Vector, Array,
)
from qdrant_orm.crud import CRUDOperations


# Define a model for recommendation example
class Product(Base):
    """Product model with vector embeddings for recommendation"""
    
    __collection__ = "products_recommend"
    
    id = Field(field_type=String(), primary_key=True)
    name = Field(field_type=String())
    category = Field(field_type=String())
    brand = Field(field_type=String())
    price = Field(field_type=Float())
    rating = Field(field_type=Float())
    tags = ArrayField(field_type=String())
    # Vector field for product embeddings
    embedding = VectorField(dimensions=384)


class Movie(Base):
    """Movie model with multiple vector fields for advanced recommendation"""
    
    __collection__ = "movies_recommend"
    
    id = Field(field_type=String(), primary_key=True)
    title = Field(field_type=String())
    genre = Field(field_type=String())
    director = Field(field_type=String())
    year = Field(field_type=Integer())
    rating = Field(field_type=Float())
    duration = Field(field_type=Integer())  # in minutes
    # Multiple vector fields for different types of embeddings
    plot_embedding = VectorField(dimensions=384)  # Plot description embedding
    visual_embedding = VectorField(dimensions=512)  # Visual features embedding


def main():
    """Main example function demonstrating recommendation functionality"""
    print("Qdrant ORM Recommendation Example")
    print("---------------------------------")
    
    # Setup connection
    print("\n1. Setting up connection to Qdrant")
    engine = QdrantEngine(url="https://57bae1dd-4983-40da-8fc4-337da62dd839.us-east4-0.gcp.cloud.qdrant.io", 
                          port=6333,
                          api_key="iiVKB5Zr8_d1GbUoLTl5-z5yHQAl4gMIpqjWbbbFWMtxfQIiZ2uLag")
    session = QdrantSession(engine)
    
    # Create collections
    print("\n2. Creating collections")
    Base.metadata.create_all(engine)
    
    # Create sample products
    print("\n3. Creating sample products")
    products = []
    
    categories = ["electronics", "clothing", "books", "home", "sports"]
    brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]
    
    for i in range(1, 21):  # Create 20 products
        # Generate random embeddings for demonstration
        embedding = np.random.rand(384).tolist()
        
        product = Product(
            id=f"prod{i}",
            name=f"Product {i}",
            category=categories[i % len(categories)],
            brand=brands[i % len(brands)],
            price=float(i * 10 + np.random.randint(1, 50)),
            rating=round(3.0 + np.random.random() * 2, 1),  # Rating between 3.0 and 5.0
            tags=[f"tag{i}", f"feature{i%3}", "quality"],
            embedding=embedding
        )
        products.append(product)
    
    # Create sample movies
    print("\n4. Creating sample movies")
    movies = []
    
    genres = ["Action", "Comedy", "Drama", "Thriller", "Sci-Fi"]
    directors = ["Director A", "Director B", "Director C", "Director D", "Director E"]
    
    for i in range(1, 16):  # Create 15 movies
        # Generate random embeddings for demonstration
        plot_embedding = np.random.rand(384).tolist()
        visual_embedding = np.random.rand(512).tolist()
        
        movie = Movie(
            id=f"movie{i}",
            title=f"Movie {i}",
            genre=genres[i % len(genres)],
            director=directors[i % len(directors)],
            year=2000 + i,
            rating=round(3.0 + np.random.random() * 2, 1),
            duration=90 + np.random.randint(1, 60),
            plot_embedding=plot_embedding,
            visual_embedding=visual_embedding
        )
        movies.append(movie)
    
    # Use bulk insert
    CRUDOperations.bulk_insert(session, products)
    CRUDOperations.bulk_insert(session, movies)
    print(f"Inserted {len(products)} products and {len(movies)} movies")
    
    # Get some items for examples
    all_products = session.query(Product).limit(10).all()
    all_movies = session.query(Movie).limit(8).all()
    print(f"Retrieved {len(all_products)} products and {len(all_movies)} movies for examples")
    
    # ===== BASIC RECOMMENDATION EXAMPLES =====
    
    # Example 1: Basic recommendation with positive IDs
    print("\n5. Basic recommendation with positive examples")
    positive_ids = [all_products[0].id, all_products[1].id]
    print(f"Using positive product IDs: {positive_ids}")
    print(f"Positive products: {[p.name for p in all_products[:2]]}")
    
    recommendations = session.query(Product).recommend(
        positive_ids=positive_ids
    ).limit(5).all()
    
    print(f"Found {len(recommendations)} product recommendations:")
    for product in recommendations:
        print(f"  - {product.name} (Category: {product.category}, Rating: {product.rating})")
    
    # Example 2: Recommendation with positive and negative IDs
    print("\n6. Recommendation with positive and negative examples")
    positive_ids = [all_products[0].id, all_products[1].id]
    negative_ids = [all_products[8].id, all_products[9].id]
    print(f"Positive IDs: {positive_ids}")
    print(f"Negative IDs: {negative_ids}")
    
    recommendations = session.query(Product).recommend(
        positive_ids=positive_ids,
        negative_ids=negative_ids
    ).limit(5).all()
    
    print(f"Found {len(recommendations)} recommendations (similar to positive, dissimilar to negative):")
    for product in recommendations:
        print(f"  - {product.name} (Category: {product.category}, Rating: {product.rating})")
    
    # ===== RECOMMENDATION WITH FILTERS =====
    
    # Example 3: Recommendation with filters
    print("\n7. Recommendation with additional filters")
    recommendations = session.query(Product).filter(
        Product.category == "electronics",
        Product.rating >= 4.0
    ).recommend(
        positive_ids=[all_products[0].id]
    ).limit(3).all()
    
    print(f"Found {len(recommendations)} electronics recommendations with rating >= 4.0:")
    for product in recommendations:
        print(f"  - {product.name} (Category: {product.category}, Rating: {product.rating}, Price: ${product.price})")
    
    # Example 4: Recommendation with price range filter
    print("\n8. Recommendation with price range filter")
    recommendations = session.query(Product).filter(
        Product.price <= 150.0
    ).recommend(
        positive_ids=[all_products[0].id, all_products[1].id]
    ).limit(4).all()
    
    print(f"Found {len(recommendations)} recommendations under $150:")
    for product in recommendations:
        print(f"  - {product.name} (Price: ${product.price}, Category: {product.category})")
    
    # ===== MULTI-VECTOR FIELD RECOMMENDATIONS =====
    
    # Example 5: Recommendation with specific vector field
    print("\n9. Recommendation with specific vector field (movies)")
    movie_positive_ids = [all_movies[0].id, all_movies[1].id]
    print(f"Using positive movie IDs: {movie_positive_ids}")
    
    # Using plot_embedding field
    recommendations = session.query(Movie).recommend(
        positive_ids=movie_positive_ids,
        using="plot_embedding"  # Specify which vector field to use
    ).limit(4).all()
    
    print(f"Found {len(recommendations)} movie recommendations using 'plot_embedding' field:")
    for movie in recommendations:
        print(f"  - {movie.title} ({movie.genre}, {movie.year}, Rating: {movie.rating})")
    
    # Using visual_embedding field
    print("\n10. Movie recommendations using 'visual_embedding' field")
    recommendations = session.query(Movie).recommend(
        positive_ids=movie_positive_ids,
        using="visual_embedding"  # Different vector field
    ).limit(4).all()
    
    print(f"Found {len(recommendations)} movie recommendations using 'visual_embedding' field:")
    for movie in recommendations:
        print(f"  - {movie.title} ({movie.genre}, {movie.year}, Rating: {movie.rating})")
    
    # ===== DIRECT VECTOR RECOMMENDATIONS =====
    
    # Example 6: Recommendation with direct vectors
    print("\n11. Recommendation with direct vectors")
    positive_vector = np.random.rand(384).tolist()
    negative_vector = np.random.rand(384).tolist()
    
    recommendations = session.query(Product).recommend(
        positive_vectors=[positive_vector],
        negative_vectors=[negative_vector]
    ).limit(5).all()
    
    print(f"Found {len(recommendations)} recommendations using direct vectors:")
    for product in recommendations:
        print(f"  - {product.name} (Category: {product.category}, Rating: {product.rating})")
    
    # ===== MIXED RECOMMENDATIONS =====
    
    # Example 7: Mixed recommendation (IDs + vectors)
    print("\n12. Mixed recommendation (IDs + vectors)")
    positive_vector = np.random.rand(384).tolist()
    
    recommendations = session.query(Product).recommend(
        positive_ids=[all_products[0].id],
        positive_vectors=[positive_vector],
        negative_ids=[all_products[5].id]
    ).limit(5).all()
    
    print(f"Found {len(recommendations)} recommendations using mixed examples:")
    for product in recommendations:
        print(f"  - {product.name} (Category: {product.category}, Rating: {product.rating})")
    
    # ===== ADVANCED QUERY CHAINING =====
    
    # Example 8: Recommendation with score threshold
    print("\n13. Recommendation with score threshold")
    recommendations = session.query(Product).recommend(
        positive_ids=[all_products[0].id, all_products[1].id]
    ).score_threshold(0.5).limit(5).all()
    
    print(f"Found {len(recommendations)} recommendations with score >= 0.5:")
    for product in recommendations:
        score = getattr(product, 'score', 'N/A')
        print(f"  - {product.name} (Category: {product.category}, Score: {score})")
    
    # Example 9: Chaining with other query methods
    print("\n14. Chaining recommendation with other query methods")
    recommendations = session.query(Product).filter(
        Product.price < 200.0
    ).recommend(
        positive_ids=[all_products[0].id]
    ).with_vectors(True).limit(3).all()
    
    print(f"Found {len(recommendations)} recommendations under $200 with vectors:")
    for product in recommendations:
        has_vectors = hasattr(product, 'embedding') and product.embedding is not None
        print(f"  - {product.name} (Price: ${product.price}, Has vectors: {has_vectors})")
    
    # Example 10: Offset and pagination
    print("\n15. Recommendation with pagination")
    # Get first page
    page_1 = session.query(Product).recommend(
        positive_ids=[all_products[0].id, all_products[1].id]
    ).limit(3).offset(0).all()
    
    # Get second page
    page_2 = session.query(Product).recommend(
        positive_ids=[all_products[0].id, all_products[1].id]
    ).limit(3).offset(3).all()
    
    print(f"Page 1 ({len(page_1)} items):")
    for product in page_1:
        print(f"  - {product.name}")
    
    print(f"Page 2 ({len(page_2)} items):")
    for product in page_2:
        print(f"  - {product.name}")
    
    # ===== PRACTICAL USE CASES =====
    
    print("\n16. Practical recommendation use cases:")
    print("==========================================")
    
    print("\nA) E-commerce product recommendations:")
    print("   - User likes products A and B → recommend similar products")
    print("   - User dislikes product C → avoid similar products")
    user_liked = [all_products[0].id, all_products[1].id]
    user_disliked = [all_products[7].id]
    
    ecommerce_recs = session.query(Product).filter(
        Product.rating >= 4.0  # Only recommend highly rated products
    ).recommend(
        positive_ids=user_liked,
        negative_ids=user_disliked
    ).limit(3).all()
    
    print(f"   Recommended products:")
    for product in ecommerce_recs:
        print(f"     - {product.name} (Rating: {product.rating}, Category: {product.category})")
    
    print("\nB) Movie streaming recommendations:")
    print("   - User likes certain genres/directors → recommend similar movies")
    user_fav_movies = [all_movies[0].id, all_movies[1].id]
    
    movie_recs = session.query(Movie).filter(
        Movie.rating >= 4.0,
        Movie.year >= 2010  # Recent movies only
    ).recommend(
        positive_ids=user_fav_movies,
        using="plot_embedding"  # Based on plot similarity
    ).limit(3).all()
    
    print(f"   Recommended movies:")
    for movie in movie_recs:
        print(f"     - {movie.title} ({movie.genre}, {movie.year}, Rating: {movie.rating})")
    
    print("\nC) Content-based filtering:")
    print("   - Use item features (embeddings) directly for recommendations")
    # Create a custom preference vector (e.g., from user's viewing history)
    preference_vector = np.random.rand(384).tolist()
    
    content_recs = session.query(Product).recommend(
        positive_vectors=[preference_vector]
    ).limit(3).all()
    
    print(f"   Content-based recommendations:")
    for product in content_recs:
        print(f"     - {product.name} (Category: {product.category})")
    
    # ===== PERFORMANCE AND BEST PRACTICES =====
    
    print("\n17. Performance tips and best practices:")
    print("=======================================")
    print("- Use filters to narrow down the search space before recommendation")
    print("- Combine multiple positive examples for better recommendations")
    print("- Use negative examples to avoid unwanted similar items")
    print("- Specify 'using' parameter when you have multiple vector fields")
    print("- Use score_threshold to filter out low-confidence recommendations")
    print("- Chain with limit() and offset() for pagination")
    print("- Use with_vectors(True) only when you need the vectors in response")
    
    print("\n18. Method chaining capabilities:")
    print("================================")
    print("session.query(Model)")
    print("  .filter(conditions)")
    print("  .recommend(positive_ids=[...], negative_ids=[...])")
    print("  .score_threshold(0.5)")
    print("  .limit(10)")
    print("  .offset(0)")
    print("  .with_payload(True)")
    print("  .with_vectors(False)")
    print("  .all()")
    
    # Count recommendations for statistics
    print("\n19. Recommendation statistics")
    total_recs = session.query(Product).recommend(
        positive_ids=[all_products[0].id]
    ).count()
    print(f"Total possible recommendations for product '{all_products[0].name}': {total_recs}")
    
    # Clean up
    print("\n20. Cleaning up")
    Base.metadata.drop_all(engine)
    print("Collections dropped")
    print("\nRecommendation example completed successfully!")


if __name__ == "__main__":
    main()
