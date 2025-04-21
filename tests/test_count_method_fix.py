import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path so we can import the qdrant_orm package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_orm.base import Base, Field, VectorField
from qdrant_orm.query import Query


class TestCountMethodFix(unittest.TestCase):
    """Test the count method filter parameter fix"""

    def setUp(self):
        # Create a mock session
        self.mock_session = MagicMock()
        self.mock_client = MagicMock()
        self.mock_session.client = self.mock_client
        
        # Create a test model class
        class TestModel(Base):
            __collection__ = "test_collection"
            
            id = Field(field_type="keyword", primary_key=True)
            name = Field(field_type="keyword")
            age = Field(field_type="integer")
            embedding = VectorField(dimensions=128)
        
        self.TestModel = TestModel

    def test_count_without_filter(self):
        """Test that count without filter works correctly"""
        # Create a query
        query = Query(self.mock_session, self.TestModel)
        
        # Execute count
        query.count()
        
        # Check that the count method was called with the correct parameters
        self.mock_client.count.assert_called_once()
        call_args = self.mock_client.count.call_args[1]
        
        # Verify only collection_name is passed (no filter)
        self.assertEqual(call_args["collection_name"], "test_collection")
        self.assertNotIn("filter", call_args)
        self.assertNotIn("count_filter", call_args)
        
    def test_count_with_filter(self):
        """Test that count with filter uses the correct parameter name"""
        # Create a query with filter
        query = Query(self.mock_session, self.TestModel)
        query.filter(self.TestModel.age > 30)
        
        # Execute count
        query.count()
        
        # Check that the count method was called with the correct parameters
        self.mock_client.count.assert_called_once()
        call_args = self.mock_client.count.call_args[1]
        
        # Verify count_filter is used instead of filter
        self.assertEqual(call_args["collection_name"], "test_collection")
        self.assertIn("count_filter", call_args)
        self.assertNotIn("filter", call_args)


if __name__ == "__main__":
    unittest.main()
