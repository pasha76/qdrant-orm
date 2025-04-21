import unittest
from unittest.mock import MagicMock, patch

from qdrant_orm.base import Base, Field, ArrayField, VectorField
from qdrant_orm.types import String, Integer, Float, Vector
from qdrant_orm.query import Query
from qdrant_orm.filters import Filter


class Document(Base):
    __collection__ = "documents"
    
    id = Field(field_type=String(), primary_key=True)
    title = Field(field_type=String())
    content = Field(field_type=String())
    rating = Field(field_type=Float())
    tags = ArrayField(field_type=String())
    keywords = ArrayField(field_type=String())
    embedding = VectorField(dimensions=128)


class TestFilterOperators(unittest.TestCase):
    
    def setUp(self):
        self.mock_session = MagicMock()
        self.mock_client = MagicMock()
        self.mock_session.client = self.mock_client
    
    def test_comparison_operators(self):
        """Test that comparison operators work correctly with Field objects"""
        # Create filters using comparison operators
        eq_filter = Document.rating == 4.5
        ne_filter = Document.rating != 3.0
        gt_filter = Document.rating > 4.0
        ge_filter = Document.rating >= 4.0
        lt_filter = Document.rating < 5.0
        le_filter = Document.rating <= 5.0
        
        # Verify that all filters are Filter objects
        self.assertIsInstance(eq_filter, Filter)
        self.assertIsInstance(ne_filter, Filter)
        self.assertIsInstance(gt_filter, Filter)
        self.assertIsInstance(ge_filter, Filter)
        self.assertIsInstance(lt_filter, Filter)
        self.assertIsInstance(le_filter, Filter)
        
        # Verify filter properties
        self.assertEqual(eq_filter.field_name, "rating")
        self.assertEqual(eq_filter.operator, "==")
        self.assertEqual(eq_filter.value, 4.5)
        
        self.assertEqual(gt_filter.field_name, "rating")
        self.assertEqual(gt_filter.operator, ">")
        self.assertEqual(gt_filter.value, 4.0)
    
    def test_array_field_operators(self):
        """Test that array field operators work correctly"""
        # Create filters using array field operators
        contains_filter = Document.tags == "tag1"  # This should use contains operator
        explicit_contains_filter = Document.tags.contains("tag1")
        contains_all_filter = Document.tags.contains_all(["tag1", "tag2"])
        contains_any_filter = Document.tags.contains_any(["tag1", "tag2"])
        
        # Verify that all filters are Filter objects
        self.assertIsInstance(contains_filter, Filter)
        self.assertIsInstance(explicit_contains_filter, Filter)
        self.assertIsInstance(contains_all_filter, Filter)
        self.assertIsInstance(contains_any_filter, Filter)
        
        # Verify filter properties
        self.assertEqual(contains_filter.field_name, "tags")
        self.assertEqual(contains_filter.operator, "contains")
        self.assertEqual(contains_filter.value, "tag1")
        
        self.assertEqual(explicit_contains_filter.field_name, "tags")
        self.assertEqual(explicit_contains_filter.operator, "contains")
        self.assertEqual(explicit_contains_filter.value, "tag1")
        
        self.assertEqual(contains_all_filter.field_name, "tags")
        self.assertEqual(contains_all_filter.operator, "contains_all")
        self.assertEqual(contains_all_filter.value, ["tag1", "tag2"])
    
    def test_query_with_comparison_operators(self):
        """Test that queries with comparison operators work correctly"""
        # Create a query with a comparison operator
        query = Query(self.mock_session, Document)
        query.filter(Document.rating > 4.0)
        
        # Mock the scroll method to return an empty result
        self.mock_client.scroll.return_value = ([], None)
        
        # Execute the query
        query.all()
        
        # Verify that scroll was called with the correct filter
        self.mock_client.scroll.assert_called_once()
        call_args = self.mock_client.scroll.call_args[1]
        
        # Verify that scroll_filter is in the parameters
        self.assertIn("scroll_filter", call_args)
    
    def test_query_with_array_field_operators(self):
        """Test that queries with array field operators work correctly"""
        # Create a query with an array field operator
        query = Query(self.mock_session, Document)
        query.filter(Document.tags == "tag1")
        
        # Mock the scroll method to return an empty result
        self.mock_client.scroll.return_value = ([], None)
        
        # Execute the query
        query.all()
        
        # Verify that scroll was called with the correct filter
        self.mock_client.scroll.assert_called_once()
        call_args = self.mock_client.scroll.call_args[1]
        
        # Verify that scroll_filter is in the parameters
        self.assertIn("scroll_filter", call_args)


if __name__ == "__main__":
    unittest.main()
