# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
- `make setup` - Create conda environment, install dependencies, and setup pre-commit hooks
- `conda activate algua` - Activate the conda environment

### Code Quality
- `make lint` - Run flake8, black --check, and isort --check
- `make format` - Format code with black and isort
- `make typecheck` - Run mypy type checking with missing imports ignored

### Testing
- `pytest -v` - Run all tests with verbose output
- `pytest --cov=. --cov-report=html --cov-report=term-missing` - Run tests with coverage
- `pytest tests/test_backtesting.py -v` - Run specific test file
- `pytest tests/unit/test_value_objects.py::TestMoney::test_addition -v` - Run specific test

### Services
- `make api` - Start FastAPI server (uvicorn api.main:app --reload --host 0.0.0.0 --port 8000)
- `make dashboard` - Start Streamlit dashboard (streamlit run dashboards/main.py --server.port 8501)

### Trading Operations
- `make backtest` - Run backtesting pipeline (python scripts/run_backtest.py)
- `make forward` - Run forward testing (python scripts/run_forward_test.py)
- `make live` - Run live trading single iteration (python scripts/run_live_trading.py)

### Utility
- `make clean` - Clean cache, temp files, and Python artifacts

## Architecture Overview

### Domain-Driven Design Structure
The codebase follows Domain-Driven Design with clear separation of concerns:

**Domain Layer (`domain/`)**:
- `entities/` - Core business entities (Strategy, Portfolio, Position, Order)
- `value_objects/` - Immutable value objects (Money, Price, Quantity, Symbol)
- Rich domain models with business logic and validation

**Infrastructure Layer (`infrastructure/`)**:
- `container.py` - Dependency injection container with autowiring capabilities
- Singleton and factory service registration patterns

**Application Services**:
- `api/` - FastAPI REST API with health checks and trading endpoints
- `backtesting/` - VectorBT Pro-based backtesting engine
- `portfolio/` - Portfolio management and performance calculation
- `risk_management/` - Position sizing and risk calculation

### Configuration Management
- Environment-based configuration in `configs/environments/` (base, development, production, testing)
- Settings factory pattern with `get_settings()` function
- Pydantic-based settings with environment variable support
- Configuration validation for production deployments

### Key Design Patterns
- **Dependency Injection**: Container-based DI with decorators (`@singleton`, `@service`, `@inject`)
- **Factory Pattern**: Settings and strategy factories
- **Repository Pattern**: Implied for data access (though not fully implemented)
- **Value Objects**: Immutable domain primitives (Money, Price, etc.)

### Trading Engine Architecture
- **Strategy Entity**: Manages strategy lifecycle, parameters, and performance metrics
- **Portfolio Entity**: Tracks positions, cash, and portfolio-level calculations
- **Position Management**: Handles position merging, P&L calculation, and allocation tracking
- **Risk Management**: Position sizing and daily loss limits

### Data Flow
1. **Data Ingestion** → Market data collection and caching
2. **Feature Engineering** → Signal generation and preprocessing  
3. **Backtesting** → Strategy validation using historical data
4. **Forward Testing** → Paper trading validation
5. **Live Trading** → Real execution via Alpaca API

### Service Dependencies
- **VectorBT Pro**: Portfolio optimization and backtesting (currently commented out)
- **Alpaca API**: Live trading and market data
- **FastAPI**: REST API framework
- **Streamlit**: Dashboard and visualization
- **PyTorch + Lightning**: ML model training (optional dependency group)
- **Optuna**: Hyperparameter optimization
- **Weights & Biases**: Experiment tracking

### Testing Strategy
- Unit tests for value objects and entities in `tests/unit/`
- Integration tests for backtesting in `tests/test_backtesting.py`
- Configuration in `pyproject.toml` with pytest options
- Coverage reporting with HTML and terminal output

### Code Style Enforcement
- **Black**: Code formatting (line length 88, Python 3.11+)
- **isort**: Import sorting (black-compatible profile)
- **flake8**: Linting
- **mypy**: Type checking (ignores missing imports)
- **pre-commit**: Automated hooks for code quality

### Environment Configuration
- Development environment includes debug mode and hot reloading
- Production environment requires additional validation
- Testing environment for isolated test runs
- Environment variables for API keys and sensitive configuration

### CLI Scripts
The platform includes several command-line scripts:
- `algua-backtest` - Run backtesting workflows
- `algua-optimize` - Hyperparameter optimization
- `algua-live` - Live trading execution

### Important Notes
- Python virtual environment created at `algua-env/` (activate with `source algua-env/bin/activate`)
- Core components implemented with pluggable interfaces for data providers and brokers
- Backtesting engine supports both VectorBT and fallback implementation
- Basic testing completed - run `python simple_test.py` to verify core functionality
- API requires Alpaca credentials for live trading functionality
- Use `make format` before committing to ensure code style compliance
- Run tests with coverage before major changes

### Implemented Components
**Pluggable Interfaces**:
- `infrastructure/interfaces.py` - Abstract interfaces for data providers and brokers
- `infrastructure/providers/yfinance_provider.py` - Yahoo Finance data provider
- `infrastructure/brokers/alpaca_broker.py` - Alpaca broker integration
- Factory patterns for easy provider/broker switching

**Enhanced Backtesting**:
- `backtesting/engine.py` - VectorBT-based engine with fallback implementation
- Supports strategy optimization and comprehensive performance metrics
- Compatible with domain entities and value objects

**Testing**:
- `simple_test.py` - Basic component tests (6/7 passing)
- `test_components.py` - Full integration tests (requires external dependencies)

### Dependencies Status
- ✅ Core Python packages: pydantic, typing-extensions
- ⚠️ Data analysis: pandas, numpy (compilation issues in Termux)
- ⚠️ Trading packages: yfinance, vectorbt, alpaca-trade-api (not yet installed)

### Next Development Steps
1. Install remaining dependencies when possible: `pip install pandas numpy yfinance vectorbt alpaca-trade-api`
2. Configure API keys in `.env` file for live data/trading
3. Implement trading strategies in `models/strategies.py`
4. Set up risk management rules
5. Test with real market data

### Large Context Analysis
When you need a bird's eye view of the codebase or help with complex architectural decisions:
- `gemini -p <prompt>` - Access Gemini agent with larger context window for comprehensive analysis
- Useful for understanding complex relationships across many files
- Good for making high-level architectural decisions that require broad context
- Has larger context than Claude but lower execution quality, so use for analysis rather than implementation