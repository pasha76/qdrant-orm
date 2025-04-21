"""
Common types and utilities for Qdrant ORM
"""
from typing import Dict, Any, List, Union


class Filter:
    """Query filtering capabilities"""
    
    def __init__(self, field_name: str, operator: str, value: Any):
        """
        Initialize a filter
        
        Args:
            field_name: Name of the field to filter on
            operator: Operator to use (==, !=, >, <, >=, <=, in, not_in, contains, contains_all, contains_any)
            value: Value to filter by
        """
        self.field_name = field_name
        self.operator = operator
        self.value = value
    
    def to_qdrant_filter(self):
        """
        This method is required for compatibility with the query system.
        The actual implementation is in Query._filter_to_qdrant_filter
        """
        # This is a placeholder - the actual conversion happens in Query._filter_to_qdrant_filter
        return self
    
    def __and__(self, other):
        """
        Combine two filters with AND logic
        
        Args:
            other: Another filter to combine with
            
        Returns:
            Combined filter
        """
        if not isinstance(other, Filter):
            raise TypeError(f"Cannot combine Filter with {type(other)}")
        
        return FilterGroup("and", [self, other])
    
    def __or__(self, other):
        """
        Combine two filters with OR logic
        
        Args:
            other: Another filter to combine with
            
        Returns:
            Combined filter
        """
        if not isinstance(other, Filter):
            raise TypeError(f"Cannot combine Filter with {type(other)}")
        
        return FilterGroup("or", [self, other])


class FilterGroup(Filter):
    """Group of filters combined with AND or OR logic"""
    
    def __init__(self, logic: str, filters: List[Filter]):
        """
        Initialize a filter group
        
        Args:
            logic: Logic to use ("and" or "or")
            filters: List of filters to combine
        """
        super().__init__("", logic, None)
        self.logic = logic
        self.filters = filters
    
    def to_qdrant_filter(self):
        """
        This method is required for compatibility with the query system.
        The actual implementation is in Query._filter_to_qdrant_filter
        """
        # This is a placeholder - the actual conversion happens in Query._filter_to_qdrant_filter
        return self
