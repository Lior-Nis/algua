"""
Main Streamlit dashboard for Algua trading platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests

# Page configuration
st.set_page_config(
    page_title="Algua Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


def load_dummy_data():
    """Load dummy data for dashboard demo."""
    # Generate dummy portfolio data
    dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='D')
    portfolio_values = 100000 + np.cumsum(np.random.randn(len(dates)) * 500)
    
    portfolio_df = pd.DataFrame({
        'date': dates,
        'value': portfolio_values,
        'daily_return': np.random.randn(len(dates)) * 0.02
    })
    
    # Generate dummy positions
    positions_df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
        'quantity': [10, 5, 15, 8, 12],
        'avg_cost': [150.0, 2500.0, 300.0, 200.0, 400.0],
        'current_price': [155.0, 2600.0, 310.0, 190.0, 420.0],
        'market_value': [1550.0, 13000.0, 4650.0, 1520.0, 5040.0],
        'unrealized_pnl': [50.0, 500.0, 150.0, -80.0, 240.0]
    })
    
    return portfolio_df, positions_df


def render_sidebar():
    """Render sidebar navigation."""
    st.sidebar.title("🚀 Algua Trading")
    st.sidebar.markdown("---")
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Portfolio Overview", "Strategy Performance", "Market Data", 
         "Risk Management", "Live Trading", "Settings"]
    )
    
    st.sidebar.markdown("---")
    
    # Quick stats
    st.sidebar.subheader("Quick Stats")
    st.sidebar.metric("Portfolio Value", "$125,760", "2.3%")
    st.sidebar.metric("Daily P&L", "$1,240", "0.98%")
    st.sidebar.metric("Active Positions", "5", "")
    
    st.sidebar.markdown("---")
    
    # Trading status
    st.sidebar.subheader("System Status")
    st.sidebar.success("✅ API Connected")
    st.sidebar.info("📊 Data Updated: 2min ago")
    st.sidebar.warning("⚠️ Paper Trading Mode")
    
    return page


def render_portfolio_overview(portfolio_df, positions_df):
    """Render portfolio overview page."""
    st.markdown('<h1 class="main-header">📊 Portfolio Overview</h1>', 
                unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Value",
            value="$125,760",
            delta="$2,340 (1.9%)"
        )
    
    with col2:
        st.metric(
            label="Available Cash",
            value="$25,000",
            delta="Available"
        )
    
    with col3:
        st.metric(
            label="Daily P&L",
            value="$1,240",
            delta="0.98%"
        )
    
    with col4:
        st.metric(
            label="Total Return",
            value="25.76%",
            delta="Since inception"
        )
    
    # Portfolio chart
    st.subheader("Portfolio Performance")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_df['date'],
        y=portfolio_df['value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Current positions
    st.subheader("Current Positions")
    
    # Add P&L color coding
    def color_pnl(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        return f'color: {color}'
    
    styled_positions = positions_df.style.applymap(
        color_pnl, subset=['unrealized_pnl']
    )
    
    st.dataframe(styled_positions, use_container_width=True)


def render_strategy_performance():
    """Render strategy performance page."""
    st.markdown('<h1 class="main-header">📈 Strategy Performance</h1>', 
                unsafe_allow_html=True)
    
    # TODO: Implement strategy performance dashboard
    st.info("🚧 Strategy performance dashboard coming soon!")
    
    # Placeholder content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Active Strategies")
        strategies = pd.DataFrame({
            'Strategy': ['Mean Reversion', 'Momentum', 'Pairs Trading'],
            'Status': ['Running', 'Paused', 'Running'],
            'Return': ['12.5%', '8.3%', '15.2%'],
            'Sharpe': [1.42, 0.98, 1.67]
        })
        st.dataframe(strategies, use_container_width=True)
    
    with col2:
        st.subheader("Strategy Allocation")
        fig = px.pie(
            values=[40, 35, 25],
            names=['Mean Reversion', 'Momentum', 'Pairs Trading'],
            title="Capital Allocation by Strategy"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_market_data():
    """Render market data page."""
    st.markdown('<h1 class="main-header">📊 Market Data</h1>', 
                unsafe_allow_html=True)
    
    # TODO: Implement real-time market data
    st.info("🚧 Real-time market data dashboard coming soon!")
    
    # Placeholder market data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    market_data = pd.DataFrame({
        'Symbol': symbols,
        'Price': [155.50, 2650.25, 315.75, 195.80, 425.60],
        'Change': ['+2.30', '+15.25', '+5.75', '-4.20', '+8.40'],
        'Change %': ['+1.5%', '+0.6%', '+1.9%', '-2.1%', '+2.0%'],
        'Volume': ['45.2M', '12.8M', '32.1M', '67.5M', '28.9M']
    })
    
    st.dataframe(market_data, use_container_width=True)


def main():
    """Main dashboard application."""
    # Load data
    portfolio_df, positions_df = load_dummy_data()
    
    # Render sidebar and get selected page
    current_page = render_sidebar()
    
    # Render selected page
    if current_page == "Portfolio Overview":
        render_portfolio_overview(portfolio_df, positions_df)
    elif current_page == "Strategy Performance":
        render_strategy_performance()
    elif current_page == "Market Data":
        render_market_data()
    else:
        st.markdown(f'<h1 class="main-header">🚧 {current_page}</h1>', 
                    unsafe_allow_html=True)
        st.info(f"{current_page} dashboard is under development!")


if __name__ == "__main__":
    main() 