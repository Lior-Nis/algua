"""
Symbol value object.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Symbol:
    """Trading symbol value object."""
    
    ticker: str
    exchange: Optional[str] = None
    
    def __post_init__(self):
        """Validate symbol."""
        if not self.ticker or not self.ticker.strip():
            raise ValueError("Symbol ticker cannot be empty")
        if len(self.ticker) > 10:
            raise ValueError("Symbol ticker too long")
        
        # Convert to uppercase for consistency
        object.__setattr__(self, 'ticker', self.ticker.upper().strip())
        if self.exchange:
            object.__setattr__(self, 'exchange', self.exchange.upper().strip())
    
    def __str__(self) -> str:
        """String representation."""
        if self.exchange:
            return f"{self.ticker}:{self.exchange}"
        return self.ticker
    
    @classmethod
    def from_string(cls, symbol_str: str) -> 'Symbol':
        """Create symbol from string representation."""
        if ':' in symbol_str:
            ticker, exchange = symbol_str.split(':', 1)
            return cls(ticker=ticker, exchange=exchange)
        return cls(ticker=symbol_str)