"""
Example usage of the Qdrant ORM framework with array fields
"""
import numpy as np
from qdrant_orm import (
    Base, Field, VectorField, ArrayField,
    QdrantEngine, QdrantSession,
    String, Integer, Float, Boolean, Vector, Array
)
from qdrant_orm.crud import CRUDOperations


# Define models with array fields
class Product(Base):
    """Product model with array fields"""
    
    __collection__ = "products"
    
    id = Field(field_type=String(), primary_key=True)
    name = Field(field_type=String())
    description = Field(field_type=String())
    price = Field(field_type=Float())
    in_stock = Field(field_type=Boolean(), default=True)
    
    # Array fields
    tags = ArrayField(field_type=String())  # Array of strings
    sizes = ArrayField(field_type=Integer())  # Array of integers
    ratings = ArrayField(field_type=Float())  # Array of floats
    
    # Vector field
    embedding = VectorField(dimensions=384)


def main():
    """Main example function"""
    print("Qdrant ORM Example with Array Fields")
    print("-----------------------------------")
    
    # Setup connection
    print("\n1. Setting up connection to Qdrant")
    engine = QdrantEngine(url="https://57bae1dd-4983-40da-8fc4-337da62dd839.us-east4-0.gcp.cloud.qdrant.io", 
                          port=6333,
                          api_key="iiVKB5Zr8_d1GbUoLTl5-z5yHQAl4gMIpqjWbbbFWMtxfQIiZ2uLag")
    session = QdrantSession(engine)
    
    # Create collections
    print("\n2. Creating collections")
    Base.metadata.create_all(engine)
    
    # Create sample data
    print("\n3. Creating sample products with array fields")
    
    # Generate random embeddings for demonstration
    product_embedding = np.random.rand(384).tolist()
    
    # Create a product with array fields
    product1 = Product(
        id="prod1",
        name="T-Shirt",
        description="Cotton t-shirt with logo",
        price=19.99,
        tags=["clothing", "casual", "summer"],  # Array of strings
        sizes=[36, 38, 40, 42],  # Array of integers
        ratings=[4.5, 4.8, 4.2, 4.7, 4.9],  # Array of floats
        embedding=product_embedding
    )
    
    # Add to session and commit
    session.add(product1)
    session.commit()
    print(f"Created product: {product1}")
    print(f"  - Tags: {product1.tags}")
    print(f"  - Sizes: {product1.sizes}")
    print(f"  - Ratings: {product1.ratings}")
    
    # Create more products with bulk insert
    print("\n4. Bulk inserting products")
    products = []
    
    product_types = ["T-Shirt", "Jeans", "Jacket", "Shoes"]
    tag_options = [
        ["clothing", "casual", "summer"],
        ["clothing", "denim", "casual"],
        ["clothing", "outerwear", "winter"],
        ["footwear", "casual", "sports"]
    ]
    
    for i in range(2, 6):
        prod_id = f"prod{i}"
        embedding = np.random.rand(384).tolist()
        product = Product(
            id=prod_id,
            name=product_types[i-2],
            description=f"Description for {product_types[i-2]}",
            price=float(i * 10) + 9.99,
            tags=tag_options[i-2],
            sizes=[36 + i*2, 38 + i*2, 40 + i*2],
            ratings=[4.0 + i/10, 4.2 + i/10, 4.5 + i/10],
            embedding=embedding
        )
        products.append(product)
    
    # Use bulk insert
    CRUDOperations.bulk_insert(session, products)
    print(f"Inserted {len(products)} products")
    
    # Basic querying
    print("\n5. Basic querying")
    
    # Get by ID
    product = session.get(Product, "prod1")
    print(f"Retrieved by ID: {product}")
    print(f"  - Tags: {product.tags}")
    print(f"  - Sizes: {product.sizes}")
    print(f"  - Ratings: {product.ratings}")
    
    # Query with filters on array fields
    print("\n6. Querying with filters on array fields")
    
    # Find products with "clothing" tag
    results = session.query(Product).filter(
        Product.tags == "clothing"
    ).all()
    print(f"Found {len(results)} products with 'clothing' tag")
    for product in results:
        print(f"  - {product.name} (Tags: {product.tags})")
    
    # Find products with size 40
    results = session.query(Product).filter(
        Product.sizes == 40,
    ).all()
    print(f"Found {len(results)} products with size 40")
    for product in results:
        print(f"  - {product.name} (Sizes: {product.sizes})")
    
    # Find products with rating above 4.5
    results = session.query(Product).filter(
      
        Product.name=="Jeans"
    ).all()
    print(f"Found {len(results)} products with rating above 4.5")
    for product in results:
        print(f"  - {product.name} (Ratings: {product.ratings})")
    
    # Update array fields
    print("\n7. Updating array fields")
    product1 = session.get(Product, "prod1")
    product1.tags.append("sale")  # Add a new tag
    product1.sizes.append(44)     # Add a new size
    session.add(product1)
    session.commit()
    
    # Verify update
    updated_product = session.get(Product, "prod1")
    print(f"Updated product tags: {updated_product.tags}")
    print(f"Updated product sizes: {updated_product.sizes}")
    
    # Clean up
    print("\n8. Cleaning up")
    Base.metadata.drop_all(engine)
    print("Collections dropped")


if __name__ == "__main__":
    main()
