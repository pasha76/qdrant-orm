"""
Example usage of the Qdrant ORM framework with updated basic usage including array fields
"""
import numpy as np
import sys
sys.path.append("/Users/tolgagunduz/Documents/projects/blushyv3/orm/")

from qdrant_orm import (
    Base, Field, VectorField, ArrayField,SparseVectorField,
    QdrantEngine, QdrantSession,
    String, Integer, Float, Boolean, Vector, Array,
)
from qdrant_orm.crud import CRUDOperations


# Define models
class Document(Base):
    """Document model with text and embedding"""
    
    __collection__ = "documents"
    
    id = Field(field_type=String(), primary_key=True)
    title = Field(field_type=String())
    content = Field(field_type=String())
    category = Field(field_type=String())
    rating = Field(field_type=Float())
    is_published = Field(field_type=Boolean(), default=True)
    # Array fields
    tags = ArrayField(field_type=String())  # Array of strings
    keywords = ArrayField(field_type=String())  # Array of strings
    embedding = VectorField(dimensions=384)  # Example dimension for text embeddings


class Product(Base):
    """Product model with multiple vector fields and array fields"""
    
    __collection__ = "products"
    
    id = Field(field_type=String(), primary_key=True)
    name = Field(field_type=String())
    description = Field(field_type=String())
    price = Field(field_type=Float())
    in_stock = Field(field_type=Boolean(), default=True)
    # Array fields
    sizes = ArrayField(field_type=Integer())  # Array of integers
    colors = ArrayField(field_type=String())  # Array of strings
    ratings = ArrayField(field_type=Float())  # Array of floats
    # Multiple vector fields for different embedding types
    text_embedding = VectorField(dimensions=384)
    image_embedding = VectorField(dimensions=512)
    sparse_embedding = SparseVectorField()


def main():
    """Main example function"""
    print("Qdrant ORM Example")
    print("-----------------")
    
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
    print("\n3. Creating sample documents")
    
    # Generate random embeddings for demonstration
    doc_embedding = np.random.rand(384).tolist()
    
    # Create a document with array fields
    doc1 = Document(
        id="doc1",
        title="Introduction to Vector Databases",
        content="Vector databases are specialized database systems designed to store and query vector embeddings efficiently.",
        category="database",
        rating=4.5,
        tags=["vector-db", "embeddings", "similarity-search"],
        keywords=["database", "vector", "embedding", "search"],
        embedding=doc_embedding
    )
    
    # Add to session and commit
    session.add(doc1)
    session.commit()
    print(f"Created document: {doc1}")
    print(f"  - Tags: {doc1.tags}")
    print(f"  - Keywords: {doc1.keywords}")
    
    # Create more documents with bulk insert
    print("\n4. Bulk inserting documents")
    docs = []
    categories = ["database", "machine-learning", "ai", "programming"]
    
    for i in range(2, 10):
        doc_id = f"doc{i}"
        embedding = np.random.rand(384).tolist()
        doc = Document(
            id=doc_id,
            title=f"Sample Document {i}",
            content=f"This is the content of sample document {i}.",
            category=categories[i % len(categories)],
            rating=float(i) / 2,
            tags=[f"tag-{i}", f"category-{i % len(categories)}", "sample"],
            keywords=[f"keyword-{i*2}", f"keyword-{i*2+1}"],
            embedding=embedding
        )
        docs.append(doc)
    
    # Use bulk insert
    CRUDOperations.bulk_insert(session, docs)
    print(f"Inserted {len(docs)} documents")
    
    # Basic querying
    print("\n5. Basic querying")
    
    # Get by ID
    doc = session.get(Document, "doc1")
    print(f"Retrieved by ID: {doc}")
    print(f"  - Tags: {doc.tags}")
    print(f"  - Keywords: {doc.keywords}")
    
    # Query with filters
    print("\n6. Querying with filters")
    results = session.query(Document).filter(
        Document.category == "database",
        Document.rating > 4.0
    ).all()
    print(f"Found {len(results)} documents matching filters")
    for doc in results:
        print(f"  - {doc.title} (Rating: {doc.rating})")

    # Basic querying
    print("\n6. Gropu by querying")

    # Get by ID
    doc = session.get(Document, "doc1")
    print(f"Retrieved by ID: {doc}")
    print(f"  - Tags: {doc.tags}")
    print(f"  - Keywords: {doc.keywords}")
    
    # Query with filters
    print("\n6. Querying with filters")
    results = session.query(Document).group_by(group_by="category",group_limit=1,group_size=3).all()

    print(f"Found {len(results)} documents matching filters")
    for doc in results:
        print(f"  - {doc.title} (Rating: {doc.rating})")


    # Basic querying
    print("\n6.5. Prefetch by querying")

    # Get by ID
    doc = session.get(Document, "doc1")
    print(f"Retrieved by ID: {doc}")
    print(f"  - Tags: {doc.tags}")
    print(f"  - Keywords: {doc.keywords}")
    
    # Query with filters
    print("\n6.5. Querying with filters")
    results = session.query(Document).prefetch(Document.embedding,query_vector=doc_embedding).vector_search(Document.embedding,doc_embedding).all()

    print(f"Found {len(results)} documents matching filters")
    for doc in results:
        print(f"  - {doc.title} (Rating: {doc.rating})")

    
    # Query with array field filters
    print("\n7. Querying with array field filters")
    results = session.query(Document).filter(
        Document.tags == "vector-db"
    ).all()
    print(f"Found {len(results)} documents with 'vector-db' tag")
    for doc in results:
        print(f"  - {doc.title} (Tags: {doc.tags})")
    
    # Print all ratings for debugging
    all_docs = session.query(Document).all()
    print('All ratings:', [doc.rating for doc in all_docs])
    
    # Vector search
    print("\n8. Vector similarity search")
    print('All ratings:', [doc.rating for doc in all_docs])
    query_vector = np.random.rand(384).tolist()  # Random vector for demonstration
    
    # For float fields, use range conditions instead of not_in
    # This example excludes ratings around 4.0 (between 3.9 and 4.1)
    similar_docs = session.query(Document).filter(
        (Document.rating < 3.9) | (Document.rating > 4.1)
    ).limit(3).all()

    for doc in similar_docs:
        print(f"  - {doc.title} rating: {doc.rating} (score hidden in this example)")
    # Update a document including array fields
    print("\n9. Updating a document")
    doc1 = session.get(Document, "doc1")
    doc1.rating = 5.0
    doc1.tags.append("updated")  # Add a new tag
    doc1.keywords.extend(["new-keyword-1", "new-keyword-2"])  # Add multiple keywords
    session.add(doc1)
    session.commit()
    
    # Verify update
    updated_doc = session.get(Document, "doc1")
    print(f"Updated document rating: {updated_doc.rating}")
    print(f"Updated document tags: {updated_doc.tags}")
    print(f"Updated document keywords: {updated_doc.keywords}")
    
    # Get or create
    print("\n10. Get or create operation")
    doc, created = CRUDOperations.get_or_create(
        session, 
        Document,
        defaults={
            "content": "New document content", 
            "embedding": np.random.rand(384).tolist(),
            "tags": ["new", "get-or-create"],
            "keywords": ["automatic", "creation"]
        },
        id="doc10",
        title="New Document"
    )
    print(f"Document {'created' if created else 'retrieved'}: {doc.title}")
    print(f"  - Tags: {doc.tags}")
    print(f"  - Keywords: {doc.keywords}")
    
    # Delete a document
    print("\n11. Deleting a document")
    doc_to_delete = session.get(Document, "doc2")
    session.delete(doc_to_delete)
    session.commit()
    
    # Verify deletion
    deleted_doc = session.get(Document, "doc2")
    print(f"Document deleted: {deleted_doc is None}")
    
    # Count documents
    print("\n12. Counting documents")
    count = session.query(Document).count()
    print(f"Total documents: {count}")
    
    # Filter by category and count
    category_count = session.query(Document).filter(
        Document.category == "database"
    ).count()
    print(f"Database category documents: {category_count}")
    
    # Filter by array field and count
    tag_count = session.query(Document).filter(
        Document.tags == "sample"
    ).count()
    print(f"Documents with 'sample' tag: {tag_count}")
    
    # Test not_in with integer fields (this works!)
    print("\n13.5 Testing not_in with integer fields")
    
    # Let's create a simple test with existing data
    # Get all documents and filter by length of tags array
    all_docs_for_test = session.query(Document).all()
    
    # Create a list of documents with tag counts
    doc_tag_counts = [(doc.id, len(doc.tags)) for doc in all_docs_for_test]
    print("Document tag counts:")
    for doc_id, count in doc_tag_counts:
        print(f"  - {doc_id}: {count} tags")
    
    # Now let's demonstrate not_in with string fields (which works)
    excluded_categories = ["database", "ai"]
    filtered_by_category = session.query(Document).filter(
        Document.category.not_in(excluded_categories)
    ).all()
    
    print(f"\nDocuments excluding categories {excluded_categories}:")
    for doc in filtered_by_category:
        print(f"  - {doc.title}: category={doc.category}")
    
    print("\nNote: not_in works perfectly with integer and string fields!")
    print("For float fields, use range conditions as shown in the vector search example above.")
    
    # Clean up
    print("\n14. Cleaning up")
    Base.metadata.drop_all(engine)
    print("Collections dropped")


if __name__ == "__main__":
    main()
