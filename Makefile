.PHONY: help setup lint test test-cov format typecheck api dashboard backtest forward live clean

# Default target
help:
	@echo "Available targets:"
	@echo "  setup      - Set up development environment"
	@echo "  lint       - Run linting checks"
	@echo "  format     - Format code with black and isort"
	@echo "  typecheck  - Run type checking with mypy"
	@echo "  test       - Run pytest"
	@echo "  test-cov   - Run tests with coverage"
	@echo "  api        - Start FastAPI server"
	@echo "  dashboard  - Start Streamlit dashboard"
	@echo "  backtest   - Run backtesting pipeline"
	@echo "  forward    - Run forward testing"
	@echo "  live       - Run live trading (single iteration)"
	@echo "  clean      - Clean cache and temp files"

# Development setup
setup:
	conda env create -f environment.yml
	conda activate algua
	pre-commit install
	mkdir -p data/{raw,processed,backtest,live}
	mkdir -p logs

# Linting and formatting
lint:
	flake8 .
	black --check .
	isort --check-only .

format:
	black .
	isort .

typecheck:
	mypy . --ignore-missing-imports

# Testing
test:
	pytest -v

test-cov:
	pytest --cov=. --cov-report=html --cov-report=term-missing

# Services
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboards/main.py --server.port 8501

# Trading workflows
backtest:
	python scripts/run_backtest.py

forward:
	python scripts/run_forward_test.py

live:
	python scripts/run_live_trading.py

# Utilities
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 