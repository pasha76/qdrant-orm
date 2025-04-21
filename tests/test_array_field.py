import unittest
import sys
import os

# Add the parent directory to the path so we can import the qdrant_orm package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_orm.base import Base, Field, ArrayField
from qdrant_orm.types import String, Integer, Float, Boolean, Array


class TestArrayField(unittest.TestCase):
    """Test the ArrayField functionality"""

    def test_array_field_initialization(self):
        """Test that ArrayField initializes correctly"""
        # Create a model with array fields
        class TestModel(Base):
            __collection__ = "test_collection"
            
            id = Field(field_type=String(), primary_key=True)
            tags = ArrayField(field_type=String())
            numbers = ArrayField(field_type=Integer())
            scores = ArrayField(field_type=Float())
            flags = ArrayField(field_type=Boolean())
        
        # Check that the fields are correctly defined
        self.assertIsInstance(TestModel._fields["tags"], ArrayField)
        self.assertIsInstance(TestModel._fields["numbers"], ArrayField)
        self.assertIsInstance(TestModel._fields["scores"], ArrayField)
        self.assertIsInstance(TestModel._fields["flags"], ArrayField)
        
        # Check that the default value is an empty list
        self.assertEqual(TestModel._fields["tags"].default, [])
        self.assertEqual(TestModel._fields["numbers"].default, [])
        self.assertEqual(TestModel._fields["scores"].default, [])
        self.assertEqual(TestModel._fields["flags"].default, [])

    def test_array_field_assignment(self):
        """Test that array values can be assigned to ArrayField"""
        # Create a model with array fields
        class TestModel(Base):
            __collection__ = "test_collection"
            
            id = Field(field_type=String(), primary_key=True)
            tags = ArrayField(field_type=String())
            numbers = ArrayField(field_type=Integer())
            scores = ArrayField(field_type=Float())
            flags = ArrayField(field_type=Boolean())
        
        # Create an instance with array values
        instance = TestModel(
            id="test1",
            tags=["tag1", "tag2", "tag3"],
            numbers=[1, 2, 3, 4, 5],
            scores=[1.1, 2.2, 3.3],
            flags=[True, False, True]
        )
        
        # Check that the values were correctly assigned
        self.assertEqual(instance.tags, ["tag1", "tag2", "tag3"])
        self.assertEqual(instance.numbers, [1, 2, 3, 4, 5])
        self.assertEqual(instance.scores, [1.1, 2.2, 3.3])
        self.assertEqual(instance.flags, [True, False, True])

    def test_array_field_validation(self):
        """Test that ArrayField validates array values"""
        # Create a model with array fields
        class TestModel(Base):
            __collection__ = "test_collection"
            
            id = Field(field_type=String(), primary_key=True)
            tags = ArrayField(field_type=String())
            numbers = ArrayField(field_type=Integer())
        
        # Create an instance with valid values
        instance = TestModel(id="test1")
        
        # Test valid assignments
        instance.tags = ["tag1", "tag2"]
        self.assertEqual(instance.tags, ["tag1", "tag2"])
        
        instance.numbers = [1, 2, 3]
        self.assertEqual(instance.numbers, [1, 2, 3])
        
        # Test invalid assignments - should raise exceptions
        with self.assertRaises(TypeError):
            instance.tags = "not_a_list"  # Not a list
            
        with self.assertRaises(ValueError):
            instance.numbers = [1, "two", 3]  # Contains non-integer

    def test_array_field_nullable(self):
        """Test that ArrayField can be nullable"""
        # Create a model with nullable and non-nullable array fields
        class TestModel(Base):
            __collection__ = "test_collection"
            
            id = Field(field_type=String(), primary_key=True)
            nullable_tags = ArrayField(field_type=String(), nullable=True)
            non_nullable_tags = ArrayField(field_type=String(), nullable=False)
        
        # Create an instance
        instance = TestModel(id="test1")
        
        # Test nullable field
        instance.nullable_tags = None
        self.assertIsNone(instance.nullable_tags)
        
        # Test non-nullable field - should raise exception
        with self.assertRaises(ValueError):
            instance.non_nullable_tags = None

    def test_array_field_custom_default(self):
        """Test that ArrayField can have custom default values"""
        # Create a model with custom default values
        class TestModel(Base):
            __collection__ = "test_collection"
            
            id = Field(field_type=String(), primary_key=True)
            tags = ArrayField(field_type=String(), default=["default1", "default2"])
            numbers = ArrayField(field_type=Integer(), default=[1, 2, 3])
        
        # Create an instance without specifying array values
        instance = TestModel(id="test1")
        
        # Check that the default values were used
        self.assertEqual(instance.tags, ["default1", "default2"])
        self.assertEqual(instance.numbers, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
