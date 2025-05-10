"""
Example usage of weighted multi-vector search in Qdrant ORM
"""
import numpy as np
import sys
sys.path.append("/Users/tolgagunduz/Documents/projects/blushyv3/orm/")
from qdrant_orm import (
    Base, Field, VectorField, SparseVectorField,
    QdrantEngine, QdrantSession,
    String, Integer, Float, Boolean, Vector
)
from qdrant_orm.crud import CRUDOperations


# Define a model with multiple vector fields
class Product(Base):
    """Product model with multiple vector fields"""
    
    __collection__ = "products"
    
    id = Field(String, primary_key=True)
    name = Field(String)
    description = Field(String)
    category = Field(String)
    price = Field(Float)
    in_stock = Field(Boolean, default=True)
    # Multiple vector fields for different embedding types
    image_embedding = VectorField(dimensions=512)
    text_embedding = VectorField(dimensions=384)
    sparse_tags = SparseVectorField()


def main():
    """Main example function demonstrating weighted multi-vector search"""
    print("Qdrant ORM Weighted Multi-Vector Search Example")
    print("----------------------------------------------")
    
    # Setup connection
    print("\n1. Setting up connection to Qdrant")
    engine = QdrantEngine(url="https://57bae1dd-4983-40da-8fc4-337da62dd839.us-east4-0.gcp.cloud.qdrant.io", 
                          port=6333,
                          api_key="iiVKB5Zr8_d1GbUoLTl5-z5yHQAl4gMIpqjWbbbFWMtxfQIiZ2uLag")
    session = QdrantSession(engine)
    
    # Create collection
    print("\n2. Creating collection")
    Base.metadata.create_all(engine)
    
    # Create sample products with both image and text embeddings
    print("\n3. Creating sample products with multiple embeddings")
    products = []
    
    # Categories for sample data
    categories = ["electronics", "clothing", "home", "books", "toys"]
    
    for i in range(1, 11):
        # Generate random embeddings for demonstration
        # In a real application, these would come from actual image and text encoders
        image_embedding = np.random.rand(512).tolist()
        text_embedding = np.random.rand(384).tolist()
        
        product = Product(
            id=f"prod{i}",
            name=f"Product {i}",
            description=f"This is a detailed description of product {i}.",
            category=categories[i % len(categories)],
            price=float(i * 10),
            image_embedding=image_embedding,
            text_embedding=text_embedding,
            sparse_tags={"indices": [0, 1, 2], "values": [1, 1, 1]}
        )
        products.append(product)
    
    # Use bulk insert
    CRUDOperations.bulk_insert(session, products)
    print(f"Inserted {len(products)} products with both image and text embeddings")
    
    # Create query vectors for search
    # In a real application, these would be generated from user input
    print("\n4. Creating query vectors for search")
    query_image_vector = np.random.rand(512).tolist()
    query_text_vector = np.random.rand(384).tolist()
    
    # Perform single vector search with image embedding
    print("\n5. Performing search with only image embedding")
    image_results = session.query(Product).vector_search(
        field=Product.image_embedding,
        query_vector=query_image_vector
    ).limit(3).all()
    
    print(f"Found {len(image_results)} products using image search:")
    for product in image_results:
        print(f"  - {product.name} (Category: {product.category}, Price: ${product.price:.2f})")
    
    # Perform single vector search with text embedding
    print("\n6. Performing search with only text embedding")
    text_results = session.query(Product).vector_search(
        field=Product.text_embedding,
        query_vector=query_text_vector
    ).limit(3).all()
    
    print(f"Found {len(text_results)} products using text search:")
    for product in text_results:
        print(f"  - {product.name} (Category: {product.category}, Price: ${product.price:.2f})")
    
    # Perform combined vector search with weights
    print("\n7. Performing combined vector search with weights")
    print("   - 70% weight for image similarity")
    print("   - 30% weight for text similarity")
    
    combined_results = session.query(Product).combined_vector_search(
        vector_fields_with_weights={
            "image_embedding": 0.9,
            "text_embedding": 0.1
        },
        query_vectors={
            "image_embedding": query_image_vector,
            "text_embedding": query_text_vector
        }
    ).limit(5).all()
    
    print(f"Found {len(combined_results)} products using combined weighted search:")
    for product in combined_results:
        print(f"  - {product.name} (Category: {product.category}, Price: ${product.price:.2f})")
    
    # Try different weight combinations
    print("\n8. Trying different weight combinations")
    
    # Equal weights (50/50)
    print("\n   a. Equal weights (50% image, 50% text)")
    equal_results = session.query(Product).combined_vector_search(
        vector_fields_with_weights={
            "image_embedding": 0.5,
            "text_embedding": 0.5
        },
        query_vectors={
            "image_embedding": query_image_vector,
            "text_embedding": query_text_vector
        }
    ).limit(5).all()
    
    print(f"Found {len(equal_results)} products with equal weights:")
    for product in equal_results:
        print(f"  - {product.name} (Category: {product.category}, Price: ${product.price:.2f})")
    
    # Text-heavy weights (20/80)
    print("\n   b. Text-heavy weights (20% image, 80% text)")
    text_heavy_results = session.query(Product).combined_vector_search(
        vector_fields_with_weights={
            "image_embedding": 0.2,
            "text_embedding": 0.8
        },
        query_vectors={
            "image_embedding": query_image_vector,
            "text_embedding": query_text_vector,
            "sparse_tags": {"indices": [0, 1, 2], "values": [1., 1., 1.]}
        }
    ).limit(5).all()
    
    print(f"Found {len(text_heavy_results)} products with text-heavy weights:")
    for product in text_heavy_results:
        print(f"  - {product.name} (Category: {product.category}, Price: ${product.price:.2f})")
    
    # Combined search with filters
    print("\n9. Combined vector search with additional filters")
    filtered_results = session.query(Product).filter(
        Product.category == "electronics",
        Product.price < 500.0
    ).combined_vector_search(
        vector_fields_with_weights={
            Product.image_embedding: 0.6,
            Product.text_embedding: 0.4
        },
        query_vectors={
            "image_embedding": query_image_vector,
            "text_embedding": query_text_vector
        },
    ).limit(5).all()
    
    print(f"Found {len(filtered_results)} electronics products under $50 with combined search:")
    for product in filtered_results:
        print(f"  - {product.name} (Category: {product.category}, Price: ${product.price:.2f})")

    print("\n9.1. Combined vector search with additional filters")
    filtered_results = session.query(Product).filter(
        Product.category == "electronics",
        Product.price < 500.0
    ).combined_vector_search(
        vector_fields_with_weights={
            Product.image_embedding: 0.8,
            Product.text_embedding: 0.,
            Product.sparse_tags: 1.
        },
        query_vectors={
            "image_embedding": query_image_vector,
            "text_embedding": query_text_vector,
            "sparse_tags": {"indices": [0, 1, 2], "values": [1., 1., 0.]}
        },
    ).limit(5).all()
    
    print(f"Found {len(filtered_results)} electronics products under $50 with combined search:")
    for product in filtered_results:
        print(f"  - {product.name} (Category: {product.category}, Price: ${product.price:.2f})")
    
    # Clean up
    print("\n10. Cleaning up")
    Base.metadata.drop_all(engine)
    print("Collection dropped")


if __name__ == "__main__":
    main()
