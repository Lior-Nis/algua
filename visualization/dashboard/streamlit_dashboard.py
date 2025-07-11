"""
Streamlit dashboard for Algua trading platform visualization.
"""

import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

try:
    import streamlit as st
    import plotly.graph_objects as go
except ImportError:
    st = None
    go = None

from domain.value_objects import Symbol
from infrastructure.interfaces import DataProviderFactory
from infrastructure.providers.simple_data_provider import SimpleDataProvider
from models.strategy_factory import StrategyFactory
from visualization.interfaces.chart_interfaces import (
    ChartData, ChartConfig, ChartType, ChartTheme, VisualizationEngine,
    IndicatorOverlay, SignalOverlay
)
from visualization.charts.tradingview_charts import (
    CandlestickChart, LineChart, PortfolioChart
)
from utils.logging import get_logger

logger = get_logger(__name__)


class AlguaDashboard:
    """Main Algua trading dashboard."""
    
    def __init__(self):
        """Initialize the dashboard."""
        if st is None:
            raise ImportError("streamlit is required for dashboard")
        
        self.viz_engine = VisualizationEngine()
        self.setup_page_config()
    
    def setup_page_config(self):
        """Configure Streamlit page."""
        st.set_page_config(
            page_title="Algua Trading Platform",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the main dashboard."""
        st.title("🏦 Algua Trading Platform Dashboard")
        st.markdown("---")
        
        # Sidebar navigation
        with st.sidebar:
            st.header("Navigation")
            page = st.selectbox(
                "Select Page",
                ["Strategy Analysis", "Portfolio Performance", "Market Data", "Backtesting"]
            )
        
        # Route to selected page
        if page == "Strategy Analysis":
            self.show_strategy_analysis()
        elif page == "Portfolio Performance":
            self.show_portfolio_performance()
        elif page == "Market Data":
            self.show_market_data()
        elif page == "Backtesting":
            self.show_backtesting()
    
    def show_strategy_analysis(self):
        """Show strategy analysis page."""
        st.header("📊 Strategy Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Strategy Configuration")
            
            # Strategy selection
            strategies = StrategyFactory.list_strategies()
            selected_strategy = st.selectbox("Select Strategy", strategies)
            
            if selected_strategy:
                strategy_info = StrategyFactory.get_strategy_info(selected_strategy)
                st.write(f"**Description:** {strategy_info['description']}")
                st.write(f"**Type:** {strategy_info['type']}")
                
                # Parameter configuration
                st.subheader("Parameters")
                params = {}
                
                if selected_strategy == 'sma_crossover':
                    params['fast_period'] = st.slider("Fast SMA Period", 5, 50, 10)
                    params['slow_period'] = st.slider("Slow SMA Period", 20, 200, 30)
                    params['min_volume'] = st.number_input("Min Volume", 0, 10000000, 100000)
                
                # Symbol selection
                symbol_input = st.text_input("Symbol", "AAPL")
                symbol = Symbol(symbol_input.upper())
                
                # Date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=100)
                
                date_range = st.slider(
                    "Analysis Period (days)",
                    30, 365, 100
                )
                start_date = end_date - timedelta(days=date_range)
        
        with col1:
            if st.button("Analyze Strategy", type="primary"):
                self._analyze_strategy(selected_strategy, params, symbol, start_date, end_date)
    
    def _analyze_strategy(self, strategy_name: str, params: Dict, symbol: Symbol, 
                         start_date: datetime, end_date: datetime):
        """Analyze strategy and show results."""
        try:
            with st.spinner("Analyzing strategy..."):
                # Get data
                provider = DataProviderFactory.create("simple")
                data = provider.get_historical_data(symbol, start_date, end_date)
                
                # Create strategy
                strategy = StrategyFactory.create_strategy(strategy_name, **params)
                
                # Generate signals
                signals = strategy.generate_signals(data)
                summary = strategy.get_signal_summary(signals)
                
                # Show strategy metrics
                st.subheader("Strategy Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Signals", summary['buy_signals'] + summary['sell_signals'])
                with col2:
                    st.metric("Buy Signals", summary['buy_signals'])
                with col3:
                    st.metric("Sell Signals", summary['sell_signals'])
                with col4:
                    st.metric("Signal Frequency", f"{summary['signal_frequency']:.2f}%")
                
                # Create candlestick chart with signals
                self._show_strategy_chart(data, signals, strategy_name, symbol)
                
                # Show signal details
                self._show_signal_details(signals, summary)
                
        except Exception as e:
            st.error(f"Error analyzing strategy: {str(e)}")
            logger.error(f"Strategy analysis error: {e}")
    
    def _show_strategy_chart(self, data: List[Dict], signals: List[Dict], 
                           strategy_name: str, symbol: Symbol):
        """Show strategy chart with signals."""
        st.subheader(f"Price Chart with {strategy_name} Signals")
        
        # Prepare chart data
        chart_data = ChartData(
            data=data,
            chart_type=ChartType.CANDLESTICK,
            title=f"{symbol} - {strategy_name}",
            metadata={'symbol': str(symbol), 'strategy': strategy_name}
        )
        
        # Create chart config
        config = ChartConfig(
            width=1000,
            height=600,
            theme=ChartTheme.TRADINGVIEW_DARK,
            show_volume=True
        )
        
        # Create candlestick chart
        chart_component = CandlestickChart(ChartType.CANDLESTICK, config)
        
        # Add SMA overlays
        if 'fast_sma' in signals[0]:
            fast_sma_data = [s['fast_sma'] for s in signals if s['fast_sma'] is not None]
            if fast_sma_data:
                fast_overlay = IndicatorOverlay(
                    name="Fast SMA",
                    data=[s['fast_sma'] for s in signals],
                    color='#00ff88',
                    line_width=2
                )
                chart_component.add_overlay(fast_overlay)
        
        if 'slow_sma' in signals[0]:
            slow_sma_data = [s['slow_sma'] for s in signals if s['slow_sma'] is not None]
            if slow_sma_data:
                slow_overlay = IndicatorOverlay(
                    name="Slow SMA",
                    data=[s['slow_sma'] for s in signals],
                    color='#ff4444',
                    line_width=2
                )
                chart_component.add_overlay(slow_overlay)
        
        # Add signal overlays
        buy_indices = [i for i, s in enumerate(signals) if s['buy_signal']]
        sell_indices = [i for i, s in enumerate(signals) if s['sell_signal']]
        prices = [item['Close'] for item in data]
        
        if buy_indices or sell_indices:
            signal_overlay = SignalOverlay(buy_indices, sell_indices, prices)
            chart_component.add_overlay(signal_overlay)
        
        # Render chart
        fig = chart_component.render(chart_data)
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_signal_details(self, signals: List[Dict], summary: Dict):
        """Show detailed signal information."""
        st.subheader("Signal Details")
        
        # Create signal DataFrame-like structure for display
        signal_data = []
        for i, signal in enumerate(signals):
            if signal['buy_signal'] or signal['sell_signal']:
                signal_data.append({
                    'Index': i,
                    'Type': 'BUY' if signal['buy_signal'] else 'SELL',
                    'Price': f"${signal['close']:.2f}",
                    'Fast SMA': f"{signal['fast_sma']:.2f}" if signal['fast_sma'] else 'N/A',
                    'Slow SMA': f"{signal['slow_sma']:.2f}" if signal['slow_sma'] else 'N/A',
                    'Volume': f"{signal['volume']:,}"
                })
        
        if signal_data:
            st.table(signal_data)
        else:
            st.info("No signals generated for this period")
    
    def show_portfolio_performance(self):
        """Show portfolio performance page."""
        st.header("💰 Portfolio Performance")
        
        # Mock portfolio data
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Portfolio Summary")
            st.metric("Total Value", "$125,430", "+$25,430")
            st.metric("Total Return", "25.43%", "+3.2%")
            st.metric("Sharpe Ratio", "1.85", "+0.15")
            st.metric("Max Drawdown", "-8.5%", "+2.1%")
            
            st.subheader("Active Positions")
            positions = [
                {"Symbol": "AAPL", "Shares": 100, "Value": "$15,000", "P&L": "+$2,000"},
                {"Symbol": "MSFT", "Shares": 50, "Value": "$15,000", "P&L": "+$1,500"},
                {"Symbol": "GOOGL", "Shares": 25, "Value": "$3,000", "P&L": "-$500"},
            ]
            
            for pos in positions:
                with st.expander(f"{pos['Symbol']} - {pos['Value']}"):
                    st.write(f"Shares: {pos['Shares']}")
                    st.write(f"P&L: {pos['P&L']}")
        
        with col1:
            # Generate mock portfolio performance data
            portfolio_data = self._generate_portfolio_data()
            
            chart_data = ChartData(
                data=portfolio_data,
                chart_type=ChartType.PORTFOLIO,
                title="Portfolio Performance Over Time"
            )
            
            config = ChartConfig(
                width=800,
                height=500,
                theme=ChartTheme.TRADINGVIEW_DARK
            )
            
            chart = self.viz_engine.create_chart(ChartType.PORTFOLIO, chart_data, config)
            st.plotly_chart(chart, use_container_width=True)
            
            # Show allocation pie chart
            self._show_allocation_chart()
    
    def _generate_portfolio_data(self) -> List[Dict]:
        """Generate mock portfolio performance data."""
        import random
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        current_date = start_date
        portfolio_value = 100000
        
        data = []
        while current_date <= end_date:
            # Random portfolio changes
            change = random.uniform(-0.02, 0.03)  # -2% to +3% daily
            portfolio_value *= (1 + change)
            
            data.append({
                'timestamp': current_date,
                'portfolio_value': portfolio_value
            })
            
            current_date += timedelta(days=1)
        
        return data
    
    def _show_allocation_chart(self):
        """Show portfolio allocation pie chart."""
        st.subheader("Portfolio Allocation")
        
        # Mock allocation data
        allocation_data = {
            'AAPL': 30,
            'MSFT': 25,
            'GOOGL': 15,
            'TSLA': 10,
            'Cash': 20
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(allocation_data.keys()),
            values=list(allocation_data.values()),
            hole=.3,
            marker_colors=['#00ff88', '#2196F3', '#ff9800', '#9c27b0', '#607d8b']
        )])
        
        fig.update_layout(
            title="Current Allocation",
            template='plotly_dark',
            paper_bgcolor='#131722',
            plot_bgcolor='#131722',
            font={'color': '#d1d4dc'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_market_data(self):
        """Show market data page."""
        st.header("📈 Market Data")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Market Overview")
            
            # Symbol input
            symbol_input = st.text_input("Symbol", "AAPL", key="market_symbol")
            
            # Time range
            time_range = st.selectbox(
                "Time Range",
                ["1D", "5D", "1M", "3M", "6M", "1Y", "2Y"]
            )
            
            # Chart type
            chart_type = st.selectbox(
                "Chart Type",
                ["Candlestick", "Line", "Volume"]
            )
            
            if st.button("Load Data", type="primary"):
                self._show_market_chart(symbol_input, time_range, chart_type)
        
        with col1:
            # Default chart
            st.info("Select a symbol and click 'Load Data' to view market data")
    
    def _show_market_chart(self, symbol_str: str, time_range: str, chart_type: str):
        """Show market data chart."""
        try:
            symbol = Symbol(symbol_str.upper())
            
            # Calculate date range
            range_days = {
                "1D": 1, "5D": 5, "1M": 30, "3M": 90,
                "6M": 180, "1Y": 365, "2Y": 730
            }
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=range_days.get(time_range, 30))
            
            # Get data
            provider = DataProviderFactory.create("simple")
            data = provider.get_historical_data(symbol, start_date, end_date)
            
            if chart_type == "Candlestick":
                chart_data = ChartData(
                    data=data,
                    chart_type=ChartType.CANDLESTICK,
                    title=f"{symbol} - {time_range}"
                )
                
                config = ChartConfig(
                    width=1000,
                    height=600,
                    theme=ChartTheme.TRADINGVIEW_DARK,
                    show_volume=True
                )
                
                chart = self.viz_engine.create_chart(ChartType.CANDLESTICK, chart_data, config)
                
            elif chart_type == "Line":
                # Convert to line chart format
                line_data = {
                    'x': [item.get('Timestamp', i) for i, item in enumerate(data)],
                    'y': [item['Close'] for item in data]
                }
                
                chart_data = ChartData(
                    data=line_data,
                    chart_type=ChartType.LINE,
                    title=f"{symbol} Close Price - {time_range}",
                    metadata={'color': '#2196F3', 'x_title': 'Date', 'y_title': 'Price ($)'}
                )
                
                config = ChartConfig(theme=ChartTheme.TRADINGVIEW_DARK)
                chart = self.viz_engine.create_chart(ChartType.LINE, chart_data, config)
            
            else:  # Volume
                chart_data = ChartData(
                    data=data,
                    chart_type=ChartType.VOLUME,
                    title=f"{symbol} Volume - {time_range}"
                )
                
                config = ChartConfig(theme=ChartTheme.TRADINGVIEW_DARK)
                chart = self.viz_engine.create_chart(ChartType.VOLUME, chart_data, config)
            
            st.plotly_chart(chart, use_container_width=True)
            
            # Show current price
            current_price = provider.get_current_price(symbol)
            st.metric(f"{symbol} Current Price", f"${current_price.value}")
            
        except Exception as e:
            st.error(f"Error loading market data: {str(e)}")
            logger.error(f"Market data error: {e}")
    
    def show_backtesting(self):
        """Show backtesting page."""
        st.header("🔬 Strategy Backtesting")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Backtest Configuration")
            
            # Strategy selection
            strategy = st.selectbox("Strategy", StrategyFactory.list_strategies())
            
            # Parameters
            if strategy == 'sma_crossover':
                fast_period = st.slider("Fast SMA", 5, 50, 10)
                slow_period = st.slider("Slow SMA", 20, 200, 30)
                params = {'fast_period': fast_period, 'slow_period': slow_period}
            
            # Symbol and date range
            symbol = st.text_input("Symbol", "AAPL")
            initial_capital = st.number_input("Initial Capital", 10000, 1000000, 100000)
            
            backtest_period = st.slider("Backtest Period (days)", 30, 365, 100)
            
            if st.button("Run Backtest", type="primary"):
                self._run_backtest(strategy, params, symbol, initial_capital, backtest_period)
        
        with col1:
            st.info("Configure backtest parameters and click 'Run Backtest'")
    
    def _run_backtest(self, strategy_name: str, params: Dict, symbol_str: str, 
                     initial_capital: float, period_days: int):
        """Run backtest and show results."""
        try:
            with st.spinner("Running backtest..."):
                # This would integrate with the actual backtesting engine
                # For now, show mock results
                
                st.subheader("Backtest Results")
                
                # Mock results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", "15.2%", "+15.2%")
                with col2:
                    st.metric("Sharpe Ratio", "1.45", "+1.45")
                with col3:
                    st.metric("Max Drawdown", "-8.5%", "-8.5%")
                with col4:
                    st.metric("Win Rate", "65%", "+65%")
                
                # Show performance chart
                performance_data = self._generate_backtest_performance()
                
                chart_data = ChartData(
                    data=performance_data,
                    chart_type=ChartType.PORTFOLIO,
                    title=f"Backtest Performance - {strategy_name}"
                )
                
                config = ChartConfig(theme=ChartTheme.TRADINGVIEW_DARK)
                chart = self.viz_engine.create_chart(ChartType.PORTFOLIO, chart_data, config)
                
                st.plotly_chart(chart, use_container_width=True)
                
                st.success("Backtest completed successfully!")
                
        except Exception as e:
            st.error(f"Backtest failed: {str(e)}")
            logger.error(f"Backtest error: {e}")
    
    def _generate_backtest_performance(self) -> List[Dict]:
        """Generate mock backtest performance data."""
        import random
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        current_date = start_date
        portfolio_value = 100000
        
        data = []
        while current_date <= end_date:
            change = random.uniform(-0.01, 0.02)  # Strategy performance
            portfolio_value *= (1 + change)
            
            data.append({
                'timestamp': current_date,
                'portfolio_value': portfolio_value
            })
            
            current_date += timedelta(days=1)
        
        return data


def main():
    """Main function to run the dashboard."""
    try:
        dashboard = AlguaDashboard()
        dashboard.run()
    except ImportError as e:
        st.error(f"Required packages not installed: {e}")
        st.info("Please install: pip install streamlit plotly")
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        logger.error(f"Dashboard error: {e}")


if __name__ == "__main__":
    main()