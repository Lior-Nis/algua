"""
Price value object.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Union


@dataclass(frozen=True)
class Price:
    """Price value object."""
    
    value: Decimal
    
    def __post_init__(self):
        """Validate and normalize price."""
        if isinstance(self.value, (int, float)):
            object.__setattr__(self, 'value', Decimal(str(self.value)))
        elif not isinstance(self.value, Decimal):
            object.__setattr__(self, 'value', Decimal(self.value))
        
        if self.value < 0:
            raise ValueError("Price cannot be negative")
    
    def __str__(self) -> str:
        """String representation."""
        return f"${self.value:.2f}"
    
    def __add__(self, other: Union['Price', Decimal, float, int]) -> 'Price':
        """Add prices."""
        if isinstance(other, Price):
            return Price(self.value + other.value)
        return Price(self.value + Decimal(str(other)))
    
    def __sub__(self, other: Union['Price', Decimal, float, int]) -> 'Price':
        """Subtract prices."""
        if isinstance(other, Price):
            return Price(self.value - other.value)
        return Price(self.value - Decimal(str(other)))
    
    def __mul__(self, other: Union[Decimal, float, int]) -> 'Price':
        """Multiply price."""
        return Price(self.value * Decimal(str(other)))
    
    def __truediv__(self, other: Union[Decimal, float, int]) -> 'Price':
        """Divide price."""
        return Price(self.value / Decimal(str(other)))
    
    def __eq__(self, other) -> bool:
        """Check equality."""
        if isinstance(other, Price):
            return self.value == other.value
        return self.value == Decimal(str(other))
    
    def __lt__(self, other) -> bool:
        """Less than comparison."""
        if isinstance(other, Price):
            return self.value < other.value
        return self.value < Decimal(str(other))
    
    def __le__(self, other) -> bool:
        """Less than or equal comparison."""
        if isinstance(other, Price):
            return self.value <= other.value
        return self.value <= Decimal(str(other))
    
    def __gt__(self, other) -> bool:
        """Greater than comparison."""
        if isinstance(other, Price):
            return self.value > other.value
        return self.value > Decimal(str(other))
    
    def __ge__(self, other) -> bool:
        """Greater than or equal comparison."""
        if isinstance(other, Price):
            return self.value >= other.value
        return self.value >= Decimal(str(other))