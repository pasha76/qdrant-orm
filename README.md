# Qdrant ORM Framework Documentation

## Overview

Qdrant ORM is a lightweight SQLAlchemy-style ORM (Object-Relational Mapping) framework for Qdrant vector database. It provides a simple and intuitive interface for defining models, performing CRUD operations, and executing vector similarity searches.

## Installation

```bash
pip install qdrant-client
```

## Quick Start

```python
from qdrant_orm import Base, Field, VectorField, QdrantEngine, QdrantSession, String, Float

# Define your model
class Document(Base):
    __collection__ = "documents"
    
    id = Field(String, primary_key=True)
    title = Field(String)
    content = Field(String)
    embedding = VectorField(dimensions=384)  # Adjust dimensions as needed

# Connect to Qdrant
engine = QdrantEngine(url="localhost", port=6333)
session = QdrantSession(engine)

# Create collection
Base.metadata.create_all(engine)

# Create and save a document
doc = Document(
    id="doc1",
    title="Sample Document",
    content="This is a sample document.",
    embedding=[0.1, 0.2, 0.3, ...]  # Your vector here
)
session.add(doc)
session.commit()

# Query documents
results = session.query(Document).filter(
    Document.title == "Sample Document"
).all()

# Vector search
similar_docs = session.query(Document).vector_search(
    Document.embedding,
    query_vector=[0.1, 0.2, 0.3, ...],
    limit=5
).all()
```

## Core Components

### Models and Fields

Models are defined as classes that inherit from `Base`. Fields are defined as class attributes using the `Field` class.

```python
class Product(Base):
    __collection__ = "products"  # Collection name in Qdrant
    
    id = Field(String, primary_key=True)
    name = Field(String)
    description = Field(String, nullable=True)
    price = Field(Float)
    embedding = VectorField(dimensions=512, distance="Cosine")
```

#### Field Types

- `String`: Text data
- `Integer`: Integer numbers
- `Float`: Floating-point numbers
- `Boolean`: True/False values
- `VectorField`: Vector embeddings with specified dimensions

#### Field Options

- `primary_key`: Marks field as the primary key (default: False)
- `nullable`: Whether the field can be null (default: True)
- `default`: Default value for the field

### Connection Management

```python
# Basic connection
engine = QdrantEngine(url="localhost", port=6333)

# Connection with authentication
engine = QdrantEngine(
    url="your-qdrant-cloud-instance.com",
    port=6333,
    api_key="your-api-key",
    https=True
)

# Create a session
session = QdrantSession(engine)
```

### CRUD Operations

#### Create

```python
# Create a new instance
doc = Document(id="doc1", title="Title", content="Content", embedding=[...])

# Add to session
session.add(doc)

# Commit changes
session.commit()
```

#### Read

```python
# Get by ID
doc = session.get(Document, "doc1")

# Query with filters
docs = session.query(Document).filter(
    Document.title == "Title",
    Document.id.in_(["doc1", "doc2"])
).all()

# Get first result
doc = session.query(Document).filter(Document.title == "Title").first()

# Count results
count = session.query(Document).filter(Document.title == "Title").count()
```

#### Update

```python
# Update an existing document
doc = session.get(Document, "doc1")
doc.title = "New Title"
session.add(doc)
session.commit()
```

#### Delete

```python
# Delete a document
doc = session.get(Document, "doc1")
session.delete(doc)
session.commit()
```

### Vector Search

```python
# Basic vector search
similar_docs = session.query(Document).vector_search(
    Document.embedding,  # Vector field to search
    query_vector=[0.1, 0.2, ...],  # Query vector
    limit=5  # Number of results
).all()

# Vector search with filters
similar_docs = session.query(Document).filter(
    Document.title == "Title"
).vector_search(
    Document.embedding,
    query_vector=[0.1, 0.2, ...],
    limit=5
).all()
```

### Weighted Multi-Vector Search

Qdrant ORM supports searching across multiple vector fields simultaneously with custom weights for each field:

```python
# Define a model with multiple vector fields
class Product(Base):
    __collection__ = "products"
    
    id = Field(String, primary_key=True)
    name = Field(String)
    image_embedding = VectorField(dimensions=512)
    text_embedding = VectorField(dimensions=384)

# Perform combined vector search with weights
results = session.query(Product).combined_vector_search(
    vector_fields_with_weights={
        Product.image_embedding: 0.7,  # 70% weight for image similarity
        Product.text_embedding: 0.3    # 30% weight for text similarity
    },
    query_vectors={
        "image_embedding": image_query_vector,
        "text_embedding": text_query_vector
    },
    limit=5
).all()
```

This allows you to balance the importance of different vector fields in your search results. For example:

- Emphasize visual similarity with higher weight on image embeddings
- Emphasize semantic similarity with higher weight on text embeddings
- Use equal weights for balanced results

You can also combine weighted vector search with filters:

```python
# Combined vector search with filters
filtered_results = session.query(Product).filter(
    Product.category == "electronics",
    Product.price < 50.0
).combined_vector_search(
    vector_fields_with_weights={
        Product.image_embedding: 0.6,
        Product.text_embedding: 0.4
    },
    query_vectors={
        "image_embedding": image_query_vector,
        "text_embedding": text_query_vector
    },
    limit=3
).all()
```

### Advanced Operations

#### Bulk Operations

```python
from qdrant_orm.crud import CRUDOperations

# Bulk insert
docs = [Document(...), Document(...), ...]
CRUDOperations.bulk_insert(session, docs)

# Bulk update
CRUDOperations.bulk_update(session, docs)

# Bulk delete
CRUDOperations.bulk_delete(session, docs)
```

#### Get or Create

```python
# Get existing or create new
doc, created = CRUDOperations.get_or_create(
    session,
    Document,
    defaults={"content": "Default content", "embedding": [...]},
    id="doc1",
    title="Title"
)
# created will be True if a new document was created, False if existing was found
```

#### Update or Create

```python
# Update existing or create new
doc, created = CRUDOperations.update_or_create(
    session,
    Document,
    defaults={"content": "Updated content", "embedding": [...]},
    id="doc1",
    title="Title"
)
```

#### Delete by Filter

```python
# Delete all matching documents
CRUDOperations.delete_by_filter(
    session,
    Document,
    Document.title == "Title"
)
```

## Multiple Vector Fields

Qdrant ORM supports models with multiple vector fields:

```python
class Product(Base):
    __collection__ = "products"
    
    id = Field(String, primary_key=True)
    name = Field(String)
    text_embedding = VectorField(dimensions=384)
    image_embedding = VectorField(dimensions=512)
```

When searching, you can either:

1. Search using a single vector field:

```python
# Search by text embedding
text_results = session.query(Product).vector_search(
    Product.text_embedding,
    query_vector=[...],
    limit=5
).all()

# Search by image embedding
image_results = session.query(Product).vector_search(
    Product.image_embedding,
    query_vector=[...],
    limit=5
).all()
```

2. Or search using multiple vector fields with weights:

```python
# Search using both embeddings with custom weights
combined_results = session.query(Product).combined_vector_search(
    vector_fields_with_weights={
        Product.image_embedding: 0.7,  # 70% weight
        Product.text_embedding: 0.3    # 30% weight
    },
    query_vectors={
        "image_embedding": image_query_vector,
        "text_embedding": text_query_vector
    },
    limit=5
).all()
```

## Best Practices

1. Always define a primary key field for your models
2. Use appropriate vector dimensions for your embeddings
3. Commit changes after adding or deleting instances
4. Use bulk operations for better performance when working with many instances
5. Use vector search with filters to narrow down results
6. When using weighted multi-vector search, experiment with different weight combinations to find the optimal balance for your use case

## Limitations

1. This is a lightweight ORM and doesn't implement all SQLAlchemy features
2. Complex joins and relationships are not supported
3. Schema migrations are not automatically handled
4. Weighted multi-vector search is implemented at the application level and may be less efficient than native database implementations

## Error Handling

```python
try:
    # Attempt operations
    doc = Document(...)
    session.add(doc)
    session.commit()
except Exception as e:
    # Handle errors
    print(f"Error: {e}")
```
