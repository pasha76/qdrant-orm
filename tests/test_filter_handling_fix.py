"""
Test file to verify the filter handling fix in the Qdrant ORM framework.

This test ensures that filters are properly converted to the format expected by Qdrant.
"""
import unittest
from unittest.mock import patch, MagicMock

from qdrant_client.http.models import Filter as QdrantFilter

from qdrant_orm import (
    Base, Field, VectorField, 
    QdrantEngine, QdrantSession,
    String, Integer, Float, Boolean, Vector
)
from qdrant_orm.filters import Filter, FilterGroup


# Define a test model
class TestDocument(Base):
    """Test document model"""
    
    __collection__ = "test_documents"
    
    id = Field(String, primary_key=True)
    title = Field(String)
    rating = Field(Float)
    tags = Field(String, is_array=True)
    embedding = VectorField(dimensions=128)


class TestFilterHandlingFix(unittest.TestCase):
    """Test case for the filter handling fix"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a mock engine and session
        self.engine = MagicMock()
        self.session = QdrantSession(self.engine)
        
        # Mock the client
        self.mock_client = MagicMock()
        self.session._get_client = MagicMock(return_value=self.mock_client)
        
        # Create a query object
        self.query = self.session.query(TestDocument)

    def test_single_filter_conversion(self):
        """Test that a single filter is properly converted to Qdrant format"""
        # Create a simple equality filter
        filter_obj = Filter("title", "==", "Test Document")
        
        # Add the filter to the query
        self.query.filter(filter_obj)
        
        # Build the Qdrant filter
        qdrant_filter = self.query._build_qdrant_filter()
        
        # Verify that the result is a QdrantFilter object
        self.assertIsInstance(qdrant_filter, QdrantFilter)
        
        # Verify that the filter has the correct structure
        self.assertTrue(hasattr(qdrant_filter, "must"))
        self.assertEqual(len(qdrant_filter.must), 1)
        
        # Verify the filter condition
        condition = qdrant_filter.must[0]
        self.assertEqual(condition["key"], "title")
        self.assertEqual(condition["match"]["value"], "Test Document")

    def test_multiple_filters_conversion(self):
        """Test that multiple filters are properly combined with AND logic"""
        # Create two filters
        filter1 = Filter("title", "==", "Test Document")
        filter2 = Filter("rating", ">", 4.0)
        
        # Add the filters to the query
        self.query.filter(filter1, filter2)
        
        # Build the Qdrant filter
        qdrant_filter = self.query._build_qdrant_filter()
        
        # Verify that the result is a QdrantFilter object
        self.assertIsInstance(qdrant_filter, QdrantFilter)
        
        # Verify that the filter has the correct structure
        self.assertTrue(hasattr(qdrant_filter, "must"))
        self.assertEqual(len(qdrant_filter.must), 2)
        
        # Verify the first filter condition
        condition1 = qdrant_filter.must[0]
        self.assertEqual(condition1["key"], "title")
        self.assertEqual(condition1["match"]["value"], "Test Document")
        
        # Verify the second filter condition
        condition2 = qdrant_filter.must[1]
        self.assertEqual(condition2["key"], "rating")
        self.assertEqual(condition2["range"]["gt"], 4.0)

    def test_filter_group_conversion(self):
        """Test that filter groups are properly converted to Qdrant format"""
        # Create filters
        filter1 = Filter("title", "==", "Test Document")
        filter2 = Filter("rating", ">", 4.0)
        
        # Create a filter group with OR logic
        filter_group = FilterGroup("or", [filter1, filter2])
        
        # Add the filter group to the query
        self.query.filter(filter_group)
        
        # Build the Qdrant filter
        qdrant_filter = self.query._build_qdrant_filter()
        
        # Verify that the result is a QdrantFilter object
        self.assertIsInstance(qdrant_filter, QdrantFilter)
        
        # Verify that the filter has the correct structure
        self.assertTrue(hasattr(qdrant_filter, "must"))
        self.assertEqual(len(qdrant_filter.must), 1)
        
        # Verify the filter group condition
        condition = qdrant_filter.must[0]
        self.assertTrue("should" in condition)
        self.assertEqual(len(condition["should"]), 2)
        
        # Verify the first condition in the group
        subcondition1 = condition["should"][0]
        self.assertEqual(subcondition1["key"], "title")
        self.assertEqual(subcondition1["match"]["value"], "Test Document")
        
        # Verify the second condition in the group
        subcondition2 = condition["should"][1]
        self.assertEqual(subcondition2["key"], "rating")
        self.assertEqual(subcondition2["range"]["gt"], 4.0)

    def test_array_field_filter_conversion(self):
        """Test that array field filters are properly converted to Qdrant format"""
        # Create an array field filter
        filter_obj = Filter("tags", "contains", "python")
        
        # Add the filter to the query
        self.query.filter(filter_obj)
        
        # Build the Qdrant filter
        qdrant_filter = self.query._build_qdrant_filter()
        
        # Verify that the result is a QdrantFilter object
        self.assertIsInstance(qdrant_filter, QdrantFilter)
        
        # Verify that the filter has the correct structure
        self.assertTrue(hasattr(qdrant_filter, "must"))
        self.assertEqual(len(qdrant_filter.must), 1)
        
        # Verify the filter condition
        condition = qdrant_filter.must[0]
        self.assertEqual(condition["key"], "tags")
        self.assertEqual(condition["match"]["value"], "python")


if __name__ == '__main__':
    unittest.main()
