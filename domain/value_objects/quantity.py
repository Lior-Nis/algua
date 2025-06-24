"""
Quantity value object.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Union


@dataclass(frozen=True)
class Quantity:
    """Quantity value object."""
    
    value: Decimal
    
    def __post_init__(self):
        """Validate and normalize quantity."""
        if isinstance(self.value, (int, float)):
            object.__setattr__(self, 'value', Decimal(str(self.value)))
        elif not isinstance(self.value, Decimal):
            object.__setattr__(self, 'value', Decimal(self.value))
        
        if self.value < 0:
            raise ValueError("Quantity cannot be negative")
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.value}"
    
    def __add__(self, other: Union['Quantity', Decimal, float, int]) -> 'Quantity':
        """Add quantities."""
        if isinstance(other, Quantity):
            return Quantity(self.value + other.value)
        return Quantity(self.value + Decimal(str(other)))
    
    def __sub__(self, other: Union['Quantity', Decimal, float, int]) -> 'Quantity':
        """Subtract quantities."""
        if isinstance(other, Quantity):
            return Quantity(self.value - other.value)
        return Quantity(self.value - Decimal(str(other)))
    
    def __mul__(self, other: Union[Decimal, float, int]) -> 'Quantity':
        """Multiply quantity."""
        return Quantity(self.value * Decimal(str(other)))
    
    def __truediv__(self, other: Union[Decimal, float, int]) -> 'Quantity':
        """Divide quantity."""
        return Quantity(self.value / Decimal(str(other)))
    
    def __eq__(self, other) -> bool:
        """Check equality."""
        if isinstance(other, Quantity):
            return self.value == other.value
        return self.value == Decimal(str(other))
    
    def __lt__(self, other) -> bool:
        """Less than comparison."""
        if isinstance(other, Quantity):
            return self.value < other.value
        return self.value < Decimal(str(other))
    
    def __le__(self, other) -> bool:
        """Less than or equal comparison."""
        if isinstance(other, Quantity):
            return self.value <= other.value
        return self.value <= Decimal(str(other))
    
    def __gt__(self, other) -> bool:
        """Greater than comparison."""
        if isinstance(other, Quantity):
            return self.value > other.value
        return self.value > Decimal(str(other))
    
    def __ge__(self, other) -> bool:
        """Greater than or equal comparison."""
        if isinstance(other, Quantity):
            return self.value >= other.value
        return self.value >= Decimal(str(other))