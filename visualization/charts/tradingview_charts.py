"""
TradingView-style chart components using Plotly.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None

from visualization.interfaces.chart_interfaces import (
    ChartComponent, ChartData, ChartConfig, ChartType,
    IndicatorOverlay, SignalOverlay, ChartFactory
)
from utils.logging import get_logger

logger = get_logger(__name__)


class CandlestickChart(ChartComponent):
    """TradingView-style candlestick chart."""
    
    def __init__(self, chart_type: ChartType, config: Optional[ChartConfig] = None):
        """Initialize candlestick chart."""
        super().__init__(chart_type, config)
        self.overlays = []
    
    def get_required_fields(self) -> List[str]:
        """Get required data fields."""
        return ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def validate_data(self, data: ChartData) -> bool:
        """Validate candlestick data."""
        try:
            raw_data = data.get_data()
            if not raw_data:
                return False
            
            required_fields = self.get_required_fields()
            
            # Check if it's a list of dictionaries
            if isinstance(raw_data, list) and raw_data:
                sample = raw_data[0]
                return all(field in sample for field in required_fields)
            
            # Could add DataFrame validation here if pandas available
            return False
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
    
    def add_overlay(self, overlay: Union[IndicatorOverlay, SignalOverlay]):
        """Add an overlay to the chart."""
        self.overlays.append(overlay)
    
    def render(self, data: ChartData) -> Any:
        """Render candlestick chart."""
        if go is None:
            raise ImportError("plotly is required for candlestick charts")
        
        raw_data = data.get_data()
        
        # Extract OHLCV data
        dates = [item.get('Timestamp', i) for i, item in enumerate(raw_data)]
        opens = [item['Open'] for item in raw_data]
        highs = [item['High'] for item in raw_data]
        lows = [item['Low'] for item in raw_data]
        closes = [item['Close'] for item in raw_data]
        volumes = [item['Volume'] for item in raw_data]
        
        # Create subplots with volume
        if self.config.show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=[data.title, 'Volume'],
                row_width=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
        
        # Add candlestick trace
        candlestick = go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name='OHLC',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff4444'
        )
        
        if self.config.show_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Add volume bars
        if self.config.show_volume:
            # Color volume bars based on price direction
            volume_colors = []
            for i in range(len(closes)):
                if i == 0:
                    volume_colors.append('#888888')
                else:
                    color = '#00ff88' if closes[i] >= closes[i-1] else '#ff4444'
                    volume_colors.append(color)
            
            volume_trace = go.Bar(
                x=dates,
                y=volumes,
                name='Volume',
                marker_color=volume_colors,
                opacity=0.6
            )
            
            fig.add_trace(volume_trace, row=2, col=1)
        
        # Add overlays
        for overlay in self.overlays:
            if isinstance(overlay, IndicatorOverlay):
                trace = overlay.to_plotly_trace(dates)
                if self.config.show_volume:
                    fig.add_trace(trace, row=1, col=1)
                else:
                    fig.add_trace(trace)
            
            elif isinstance(overlay, SignalOverlay):
                traces = overlay.to_plotly_traces(dates)
                for trace in traces:
                    if self.config.show_volume:
                        fig.add_trace(trace, row=1, col=1)
                    else:
                        fig.add_trace(trace)
        
        # Update layout
        layout = self.config.get_plotly_layout()
        
        # TradingView-specific styling
        layout.update({
            'title': data.title,
            'xaxis_rangeslider_visible': False,  # Hide range slider
            'dragmode': 'pan',
            'selectdirection': 'horizontal',
        })
        
        if self.config.show_volume:
            layout.update({
                'xaxis2_title': 'Date',
                'yaxis_title': 'Price',
                'yaxis2_title': 'Volume',
                'xaxis2_type': 'date',
            })
        else:
            layout.update({
                'xaxis_title': 'Date',
                'yaxis_title': 'Price',
                'xaxis_type': 'date',
            })
        
        fig.update_layout(layout)
        
        return fig


class LineChart(ChartComponent):
    """Simple line chart for indicators and portfolio performance."""
    
    def get_required_fields(self) -> List[str]:
        """Get required data fields."""
        return ['x', 'y']
    
    def validate_data(self, data: ChartData) -> bool:
        """Validate line chart data."""
        try:
            raw_data = data.get_data()
            if not raw_data:
                return False
            
            # Check for x, y data structure
            if isinstance(raw_data, dict):
                return 'x' in raw_data and 'y' in raw_data
            
            # Check for list of dictionaries
            if isinstance(raw_data, list) and raw_data:
                sample = raw_data[0]
                return 'x' in sample and 'y' in sample
            
            return False
            
        except Exception as e:
            logger.error(f"Line chart data validation failed: {e}")
            return False
    
    def render(self, data: ChartData) -> Any:
        """Render line chart."""
        if go is None:
            raise ImportError("plotly is required for line charts")
        
        raw_data = data.get_data()
        
        # Extract x, y data
        if isinstance(raw_data, dict):
            x_data = raw_data['x']
            y_data = raw_data['y']
        else:
            x_data = [item['x'] for item in raw_data]
            y_data = [item['y'] for item in raw_data]
        
        # Create line trace
        line_trace = go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            name=data.title,
            line=dict(
                color=data.get_metadata('color', '#2196F3'),
                width=data.get_metadata('line_width', 2)
            )
        )
        
        fig = go.Figure(data=[line_trace])
        
        # Update layout
        layout = self.config.get_plotly_layout()
        layout.update({
            'title': data.title,
            'xaxis_title': data.get_metadata('x_title', 'X'),
            'yaxis_title': data.get_metadata('y_title', 'Y'),
        })
        
        fig.update_layout(layout)
        
        return fig


class PortfolioChart(ChartComponent):
    """Portfolio performance chart."""
    
    def get_required_fields(self) -> List[str]:
        """Get required data fields."""
        return ['timestamp', 'portfolio_value']
    
    def validate_data(self, data: ChartData) -> bool:
        """Validate portfolio data."""
        try:
            raw_data = data.get_data()
            if not raw_data:
                return False
            
            required_fields = self.get_required_fields()
            
            if isinstance(raw_data, list) and raw_data:
                sample = raw_data[0]
                return all(field in sample for field in required_fields)
            
            return False
            
        except Exception as e:
            logger.error(f"Portfolio chart data validation failed: {e}")
            return False
    
    def render(self, data: ChartData) -> Any:
        """Render portfolio performance chart."""
        if go is None:
            raise ImportError("plotly is required for portfolio charts")
        
        raw_data = data.get_data()
        
        # Extract portfolio data
        timestamps = [item['timestamp'] for item in raw_data]
        values = [item['portfolio_value'] for item in raw_data]
        
        # Calculate returns
        returns = []
        initial_value = values[0] if values else 100000
        for value in values:
            return_pct = ((value - initial_value) / initial_value) * 100
            returns.append(return_pct)
        
        # Create portfolio value trace
        portfolio_trace = go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#2196F3', width=3),
            fill='tonexty' if data.get_metadata('fill', False) else None
        )
        
        # Create returns trace (secondary y-axis)
        returns_trace = go.Scatter(
            x=timestamps,
            y=returns,
            mode='lines',
            name='Returns (%)',
            line=dict(color='#ff9800', width=2),
            yaxis='y2',
            visible='legendonly'  # Hidden by default
        )
        
        fig = go.Figure()
        fig.add_trace(portfolio_trace)
        fig.add_trace(returns_trace)
        
        # Update layout with dual y-axis
        layout = self.config.get_plotly_layout()
        layout.update({
            'title': data.title or 'Portfolio Performance',
            'xaxis_title': 'Date',
            'yaxis_title': 'Portfolio Value ($)',
            'yaxis2': dict(
                title='Returns (%)',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            'hovermode': 'x unified'
        })
        
        fig.update_layout(layout)
        
        return fig


class VolumeChart(ChartComponent):
    """Volume chart component."""
    
    def get_required_fields(self) -> List[str]:
        """Get required data fields."""
        return ['Volume']
    
    def validate_data(self, data: ChartData) -> bool:
        """Validate volume data."""
        try:
            raw_data = data.get_data()
            if not raw_data:
                return False
            
            if isinstance(raw_data, list) and raw_data:
                sample = raw_data[0]
                return 'Volume' in sample
            
            return False
            
        except Exception as e:
            logger.error(f"Volume chart data validation failed: {e}")
            return False
    
    def render(self, data: ChartData) -> Any:
        """Render volume chart."""
        if go is None:
            raise ImportError("plotly is required for volume charts")
        
        raw_data = data.get_data()
        
        # Extract volume data
        dates = [item.get('Timestamp', i) for i, item in enumerate(raw_data)]
        volumes = [item['Volume'] for item in raw_data]
        
        # Color based on price movement if Close price available
        colors = []
        if 'Close' in raw_data[0]:
            closes = [item['Close'] for item in raw_data]
            for i in range(len(closes)):
                if i == 0:
                    colors.append('#888888')
                else:
                    color = '#00ff88' if closes[i] >= closes[i-1] else '#ff4444'
                    colors.append(color)
        else:
            colors = ['#888888'] * len(volumes)
        
        volume_trace = go.Bar(
            x=dates,
            y=volumes,
            name='Volume',
            marker_color=colors,
            opacity=0.7
        )
        
        fig = go.Figure(data=[volume_trace])
        
        # Update layout
        layout = self.config.get_plotly_layout()
        layout.update({
            'title': data.title or 'Volume',
            'xaxis_title': 'Date',
            'yaxis_title': 'Volume',
            'xaxis_type': 'date' if 'Timestamp' in raw_data[0] else 'linear'
        })
        
        fig.update_layout(layout)
        
        return fig


# Register chart components
ChartFactory.register_component(ChartType.CANDLESTICK, CandlestickChart)
ChartFactory.register_component(ChartType.LINE, LineChart)
ChartFactory.register_component(ChartType.PORTFOLIO, PortfolioChart)
ChartFactory.register_component(ChartType.VOLUME, VolumeChart)