"""
Money value object.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Union


@dataclass(frozen=True)
class Money:
    """Money value object with currency."""
    
    amount: Decimal
    currency: str = "USD"
    
    def __post_init__(self):
        """Validate and normalize money."""
        if isinstance(self.amount, (int, float)):
            object.__setattr__(self, 'amount', Decimal(str(self.amount)))
        elif not isinstance(self.amount, Decimal):
            object.__setattr__(self, 'amount', Decimal(self.amount))
        
        object.__setattr__(self, 'currency', self.currency.upper())
        
        if len(self.currency) != 3:
            raise ValueError("Currency code must be 3 characters")
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.amount:.2f} {self.currency}"
    
    def __add__(self, other: 'Money') -> 'Money':
        """Add money (same currency only)."""
        if not isinstance(other, Money):
            raise TypeError("Can only add Money to Money")
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return Money(self.amount + other.amount, self.currency)
    
    def __sub__(self, other: 'Money') -> 'Money':
        """Subtract money (same currency only)."""
        if not isinstance(other, Money):
            raise TypeError("Can only subtract Money from Money")
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract {other.currency} from {self.currency}")
        return Money(self.amount - other.amount, self.currency)
    
    def __mul__(self, other: Union[Decimal, float, int]) -> 'Money':
        """Multiply money by a number."""
        return Money(self.amount * Decimal(str(other)), self.currency)
    
    def __truediv__(self, other: Union[Decimal, float, int]) -> 'Money':
        """Divide money by a number."""
        return Money(self.amount / Decimal(str(other)), self.currency)
    
    def __eq__(self, other) -> bool:
        """Check equality."""
        if not isinstance(other, Money):
            return False
        return self.amount == other.amount and self.currency == other.currency
    
    def __lt__(self, other: 'Money') -> bool:
        """Less than comparison (same currency only)."""
        if not isinstance(other, Money):
            raise TypeError("Can only compare Money to Money")
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare {self.currency} and {other.currency}")
        return self.amount < other.amount
    
    def __le__(self, other: 'Money') -> bool:
        """Less than or equal comparison."""
        return self < other or self == other
    
    def __gt__(self, other: 'Money') -> bool:
        """Greater than comparison."""
        if not isinstance(other, Money):
            raise TypeError("Can only compare Money to Money")
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare {self.currency} and {other.currency}")
        return self.amount > other.amount
    
    def __ge__(self, other: 'Money') -> bool:
        """Greater than or equal comparison."""
        return self > other or self == other
    
    @property
    def is_positive(self) -> bool:
        """Check if amount is positive."""
        return self.amount > 0
    
    @property
    def is_negative(self) -> bool:
        """Check if amount is negative."""
        return self.amount < 0
    
    @property
    def is_zero(self) -> bool:
        """Check if amount is zero."""
        return self.amount == 0