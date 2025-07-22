"""Test all filter operators in qdrant_orm"""

import pytest
from qdrant_orm.query import Query
from qdrant_orm.filters import Filter


class TestFilterOperators:
    """Test various filter operators"""
    
    def test_basic_operators(self):
        """Test basic equality and comparison operators"""
        # Test ==
        f = Filter("name", "==", "test")
        assert f.operator == "=="
        
        # Test !=
        f = Filter("name", "!=", "test")
        assert f.operator == "!="
        
        # Test comparison operators
        for op in [">", ">=", "<", "<="]:
            f = Filter("price", op, 100)
            assert f.operator == op
    
    def test_list_operators(self):
        """Test in and not_in operators"""
        # Test in
        f = Filter("category", "in", ["laptop", "desktop"])
        assert f.operator == "in"
        
        # Test not_in
        f = Filter("category", "not_in", ["tablet", "phone"])
        assert f.operator == "not_in"
    
    def test_contains_operators(self):
        """Test contains operators"""
        # Test contains
        f = Filter("tags", "contains", "electronics")
        assert f.operator == "contains"
        
        # Test contains_any
        f = Filter("tags", "contains_any", ["laptop", "desktop"])
        assert f.operator == "contains_any"
        
        # Test contains_all
        f = Filter("tags", "contains_all", ["laptop", "gaming"])
        assert f.operator == "contains_all"
    
    def test_special_operators(self):
        """Test special operators like is_empty, is_null, text_match"""
        # Test is_empty
        f = Filter("description", "is_empty", True)
        assert f.operator == "is_empty"
        
        # Test is_null
        f = Filter("discount", "is_null", True)
        assert f.operator == "is_null"
        
        # Test text_match
        f = Filter("description", "text_match", "gaming laptop")
        assert f.operator == "text_match"
        
        # Test values_count
        f = Filter("tags", "values_count", {"gt": 2})
        assert f.operator == "values_count"


if __name__ == "__main__":
    # Run some basic tests
    test = TestFilterOperators()
    test.test_basic_operators()
    test.test_list_operators()
    test.test_contains_operators()
    test.test_special_operators()
    print("All filter operator tests passed!")
