"""
Data types for Qdrant ORM
"""
from typing import List, Optional, Union, Any, Type, TypeVar, Generic


class DataType:
    """Base class for all data types"""
    
    def __init__(self, nullable: bool = True):
        self.nullable = nullable
    
    def validate(self, value: Any) -> bool:
        """Validate a value against this type"""
        if value is None:
            return self.nullable
        return True
    
    def to_qdrant_type(self) -> str:
        """Convert to Qdrant type name"""
        raise NotImplementedError("Subclasses must implement to_qdrant_type")


class Array(DataType):
    """Array data type that wraps another data type"""
    
    def __init__(self, base_type: DataType, nullable: bool = True):
        """
        Initialize an array type
        
        Args:
            base_type: The data type of elements in the array
            nullable: Whether the array itself can be null
        """
        super().__init__(nullable=nullable)
        self.base_type = base_type
    
    def validate(self, value: Any) -> bool:
        """Validate that value is an array with elements of the base type"""
        if not super().validate(value):
            return False
        
        if value is None:
            return True
        
        if not isinstance(value, list):
            return False
        
        # Validate each element in the array
        return all(self.base_type.validate(item) for item in value)
    
    def to_qdrant_type(self) -> str:
        """
        Convert to Qdrant type name
        
        Note: Qdrant handles arrays natively, so we return the base type's name
        """
        return self.base_type.to_qdrant_type()


class String(DataType):
    """String data type"""
    
    def validate(self, value: Any) -> bool:
        if not super().validate(value):
            return False
        return value is None or isinstance(value, str)
    
    def to_qdrant_type(self) -> str:
        return "keyword"


class Integer(DataType):
    """Integer data type"""
    
    def validate(self, value: Any) -> bool:
        if not super().validate(value):
            return False
        return value is None or isinstance(value, int)
    
    def to_qdrant_type(self) -> str:
        return "integer"


class Float(DataType):
    """Float data type"""
    
    def validate(self, value: Any) -> bool:
        if not super().validate(value):
            return False
        return value is None or isinstance(value, (int, float))
    
    def to_qdrant_type(self) -> str:
        return "float"


class Boolean(DataType):
    """Boolean data type"""
    
    def validate(self, value: Any) -> bool:
        if not super().validate(value):
            return False
        return value is None or isinstance(value, bool)
    
    def to_qdrant_type(self) -> str:
        return "bool"


class Vector(DataType):
    """Vector data type"""
    
    def __init__(self, dimensions: int, distance: str = "Cosine", nullable: bool = True):
        super().__init__(nullable=nullable)
        self.dimensions = dimensions
        self.distance = distance
    
    def validate(self, value: Any) -> bool:
        if not super().validate(value):
            return False
        if value is None:
            return True
        if not isinstance(value, list):
            return False
        if len(value) != self.dimensions:
            return False
        return all(isinstance(x, (int, float)) for x in value)
    
    def to_qdrant_type(self) -> str:
        return "vector"
