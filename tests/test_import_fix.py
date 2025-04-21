"""
Test file to verify the import fix for the filters module.

This test ensures that the import statements in query.py correctly
reference the classes that exist in filters.py.
"""
import unittest
from unittest.mock import patch, MagicMock

# Test the imports
from qdrant_orm.query import Query
from qdrant_orm.filters import Filter, FilterGroup


class TestImportFix(unittest.TestCase):
    """Test case for the import fix"""
    
    def test_imports_work_correctly(self):
        """Test that the imports work correctly"""
        # If we got this far without an ImportError, the test passes
        self.assertTrue(True)
    
    def test_filter_class_exists(self):
        """Test that the Filter class exists and can be instantiated"""
        filter_obj = Filter("test_field", "==", "test_value")
        self.assertIsInstance(filter_obj, Filter)
    
    def test_filter_group_class_exists(self):
        """Test that the FilterGroup class exists and can be instantiated"""
        filter1 = Filter("field1", "==", "value1")
        filter2 = Filter("field2", "==", "value2")
        filter_group = FilterGroup("and", [filter1, filter2])
        self.assertIsInstance(filter_group, FilterGroup)


if __name__ == '__main__':
    unittest.main()
