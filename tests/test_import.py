"""
Test script to verify that ArrayField and Array can be imported correctly
"""

# Try importing the classes from the package
try:
    from qdrant_orm import Base, Field, VectorField, ArrayField, String, Integer, Float, Boolean, Vector, Array
    print("✅ Successfully imported all classes including ArrayField and Array")
    
    # Create a simple model with array fields to verify further
    class TestModel(Base):
        __collection__ = "test_collection"
        
        id = Field(field_type=String(), primary_key=True)
        name = Field(field_type=String())
        tags = ArrayField(field_type=String())
        numbers = ArrayField(field_type=Integer())
    
    # Create an instance
    instance = TestModel(
        id="test1",
        name="Test Model",
        tags=["tag1", "tag2"],
        numbers=[1, 2, 3]
    )
    
    print(f"✅ Successfully created model instance with array fields: {instance}")
    print(f"   - Tags: {instance.tags}")
    print(f"   - Numbers: {instance.numbers}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
