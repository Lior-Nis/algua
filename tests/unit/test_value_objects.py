"""
Unit tests for domain value objects.
"""

import pytest
from decimal import Decimal

from domain.value_objects import Symbol, Price, Quantity, Money


class TestSymbol:
    """Test Symbol value object."""
    
    def test_symbol_creation(self):
        """Test basic symbol creation."""
        symbol = Symbol("AAPL", "NASDAQ")
        assert symbol.ticker == "AAPL"
        assert symbol.exchange == "NASDAQ"
        assert str(symbol) == "AAPL:NASDAQ"
    
    def test_symbol_without_exchange(self):
        """Test symbol without exchange."""
        symbol = Symbol("AAPL")
        assert symbol.ticker == "AAPL"
        assert symbol.exchange is None
        assert str(symbol) == "AAPL"
    
    def test_symbol_case_normalization(self):
        """Test symbol case normalization."""
        symbol = Symbol("aapl", "nasdaq")
        assert symbol.ticker == "AAPL"
        assert symbol.exchange == "NASDAQ"
    
    def test_symbol_from_string(self):
        """Test creating symbol from string."""
        symbol = Symbol.from_string("AAPL:NASDAQ")
        assert symbol.ticker == "AAPL"
        assert symbol.exchange == "NASDAQ"
        
        symbol = Symbol.from_string("GOOGL")
        assert symbol.ticker == "GOOGL"
        assert symbol.exchange is None
    
    def test_symbol_validation(self):
        """Test symbol validation."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Symbol("")
            
        with pytest.raises(ValueError, match="too long"):
            Symbol("VERYLONGTICKER")
    
    def test_symbol_immutability(self):
        """Test that symbols are immutable."""
        symbol = Symbol("AAPL")
        with pytest.raises(AttributeError):
            symbol.ticker = "GOOGL"


class TestPrice:
    """Test Price value object."""
    
    def test_price_creation(self):
        """Test basic price creation."""
        price = Price(Decimal("100.50"))
        assert price.value == Decimal("100.50")
        assert str(price) == "$100.50"
    
    def test_price_from_float(self):
        """Test price creation from float."""
        price = Price(100.50)
        assert price.value == Decimal("100.50")
    
    def test_price_from_int(self):
        """Test price creation from int."""
        price = Price(100)
        assert price.value == Decimal("100")
    
    def test_price_validation(self):
        """Test price validation."""
        with pytest.raises(ValueError, match="cannot be negative"):
            Price(-10.50)
    
    def test_price_arithmetic(self):
        """Test price arithmetic operations."""
        price1 = Price(100.50)
        price2 = Price(50.25)
        
        # Addition
        result = price1 + price2
        assert result.value == Decimal("150.75")
        
        # Subtraction
        result = price1 - price2
        assert result.value == Decimal("50.25")
        
        # Multiplication
        result = price1 * 2
        assert result.value == Decimal("201.00")
        
        # Division
        result = price1 / 2
        assert result.value == Decimal("50.25")
    
    def test_price_comparison(self):
        """Test price comparison operations."""
        price1 = Price(100.50)
        price2 = Price(50.25)
        price3 = Price(100.50)
        
        assert price1 > price2
        assert price2 < price1
        assert price1 == price3
        assert price1 >= price3
        assert price2 <= price1


class TestQuantity:
    """Test Quantity value object."""
    
    def test_quantity_creation(self):
        """Test basic quantity creation."""
        qty = Quantity(Decimal("100"))
        assert qty.value == Decimal("100")
        assert str(qty) == "100"
    
    def test_quantity_validation(self):
        """Test quantity validation."""
        with pytest.raises(ValueError, match="cannot be negative"):
            Quantity(-100)
    
    def test_quantity_arithmetic(self):
        """Test quantity arithmetic operations."""
        qty1 = Quantity(100)
        qty2 = Quantity(50)
        
        # Addition
        result = qty1 + qty2
        assert result.value == Decimal("150")
        
        # Subtraction
        result = qty1 - qty2
        assert result.value == Decimal("50")
        
        # Multiplication
        result = qty1 * 2
        assert result.value == Decimal("200")


class TestMoney:
    """Test Money value object."""
    
    def test_money_creation(self):
        """Test basic money creation."""
        money = Money(Decimal("1000.50"), "USD")
        assert money.amount == Decimal("1000.50")
        assert money.currency == "USD"
        assert str(money) == "1000.50 USD"
    
    def test_money_default_currency(self):
        """Test money with default currency."""
        money = Money(1000)
        assert money.currency == "USD"
    
    def test_money_currency_normalization(self):
        """Test currency code normalization."""
        money = Money(1000, "usd")
        assert money.currency == "USD"
    
    def test_money_validation(self):
        """Test money validation."""
        with pytest.raises(ValueError, match="must be 3 characters"):
            Money(1000, "US")
    
    def test_money_arithmetic_same_currency(self):
        """Test money arithmetic with same currency."""
        money1 = Money(1000, "USD")
        money2 = Money(500, "USD")
        
        # Addition
        result = money1 + money2
        assert result.amount == Decimal("1500")
        assert result.currency == "USD"
        
        # Subtraction
        result = money1 - money2
        assert result.amount == Decimal("500")
        assert result.currency == "USD"
    
    def test_money_arithmetic_different_currency(self):
        """Test money arithmetic with different currencies."""
        money1 = Money(1000, "USD")
        money2 = Money(500, "EUR")
        
        with pytest.raises(ValueError, match="Cannot add USD and EUR"):
            money1 + money2
        
        with pytest.raises(ValueError, match="Cannot subtract EUR from USD"):
            money1 - money2
    
    def test_money_multiplication_division(self):
        """Test money multiplication and division."""
        money = Money(1000, "USD")
        
        # Multiplication
        result = money * 2
        assert result.amount == Decimal("2000")
        assert result.currency == "USD"
        
        # Division
        result = money / 2
        assert result.amount == Decimal("500")
        assert result.currency == "USD"
    
    def test_money_comparison_same_currency(self):
        """Test money comparison with same currency."""
        money1 = Money(1000, "USD")
        money2 = Money(500, "USD")
        money3 = Money(1000, "USD")
        
        assert money1 > money2
        assert money2 < money1
        assert money1 == money3
    
    def test_money_comparison_different_currency(self):
        """Test money comparison with different currencies."""
        money1 = Money(1000, "USD")
        money2 = Money(500, "EUR")
        
        with pytest.raises(ValueError, match="Cannot compare USD and EUR"):
            money1 > money2
    
    def test_money_properties(self):
        """Test money boolean properties."""
        positive_money = Money(1000, "USD")
        negative_money = Money(-500, "USD")
        zero_money = Money(0, "USD")
        
        assert positive_money.is_positive
        assert not positive_money.is_negative
        assert not positive_money.is_zero
        
        assert not negative_money.is_positive
        assert negative_money.is_negative
        assert not negative_money.is_zero
        
        assert not zero_money.is_positive
        assert not zero_money.is_negative
        assert zero_money.is_zero