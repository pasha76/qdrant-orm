"""
Simple test to verify the recommend method exists
"""
import sys
sys.path.append("/Users/tolgagunduz/Documents/projects/blushyv3/orm/")

# Force reload of modules to ensure latest changes
import importlib
try:
    import qdrant_orm.query
    importlib.reload(qdrant_orm.query)
    print("✅ Reloaded qdrant_orm.query module")
except Exception as e:
    print(f"⚠️  Could not reload module: {e}")

from qdrant_orm import QdrantEngine, QdrantSession, Base, Field, VectorField, String
from qdrant_orm.query import Query


class TestModel(Base):
    __collection__ = "test_recommend"
    id = Field(field_type=String(), primary_key=True)
    embedding = VectorField(dimensions=384)


def test_recommend_method():
    """Test if the recommend method exists"""
    print("Testing recommend method existence...")
    
    # Setup minimal engine and session
    engine = QdrantEngine(url="https://57bae1dd-4983-40da-8fc4-337da62dd839.us-east4-0.gcp.cloud.qdrant.io", 
                          port=6333,
                          api_key="iiVKB5Zr8_d1GbUoLTl5-z5yHQAl4gMIpqjWbbbFWMtxfQIiZ2uLag")
    session = QdrantSession(engine)
    
    # Create query object
    query = session.query(TestModel)
    
    # Check if recommend method exists
    print(f"Query object type: {type(query)}")
    
    # List all public methods
    public_methods = [method for method in dir(query) if not method.startswith('_')]
    print(f"Query object methods: {public_methods}")
    
    # Check specifically for recommend
    has_recommend = hasattr(query, 'recommend')
    print(f"Has recommend method: {has_recommend}")
    
    if has_recommend:
        print("✅ recommend method exists!")
        # Try to call it
        try:
            query_with_recommend = query.recommend(positive_ids=["test1"])
            print(f"✅ recommend method callable, returns: {type(query_with_recommend)}")
            print("✅ All tests passed!")
        except Exception as e:
            print(f"❌ Error calling recommend method: {e}")
    else:
        print("❌ recommend method does NOT exist!")
        print("Available methods:")
        for method in public_methods:
            print(f"  - {method}")
        
    # Also check the Query class directly
    query_class_methods = [method for method in dir(Query) if not method.startswith('_')]
    print(f"\nQuery class methods: {query_class_methods}")
    print(f"Query class has recommend: {hasattr(Query, 'recommend')}")


if __name__ == "__main__":
    test_recommend_method()
