"""
Chart interfaces for the visualization system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    go = None
    px = None


class ChartType(Enum):
    """Supported chart types."""
    CANDLESTICK = "candlestick"
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    VOLUME = "volume"
    INDICATOR = "indicator"
    PORTFOLIO = "portfolio"
    SIGNALS = "signals"


class ChartTheme(Enum):
    """Chart themes."""
    TRADINGVIEW_DARK = "tradingview_dark"
    TRADINGVIEW_LIGHT = "tradingview_light"
    PLOTLY_DARK = "plotly_dark"
    PLOTLY_WHITE = "plotly_white"
    CUSTOM = "custom"


class ChartData:
    """Container for chart data."""
    
    def __init__(self, 
                 data: Union[List[Dict], Any],
                 chart_type: ChartType,
                 title: str = "",
                 metadata: Optional[Dict] = None):
        """
        Initialize chart data.
        
        Args:
            data: The raw data (list of dicts or DataFrame)
            chart_type: Type of chart
            title: Chart title
            metadata: Additional metadata
        """
        self.data = data
        self.chart_type = chart_type
        self.title = title
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def get_data(self) -> Union[List[Dict], Any]:
        """Get the chart data."""
        return self.data
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = value


class ChartConfig:
    """Chart configuration settings."""
    
    def __init__(self,
                 width: int = 1200,
                 height: int = 600,
                 theme: ChartTheme = ChartTheme.TRADINGVIEW_DARK,
                 show_volume: bool = True,
                 show_grid: bool = True,
                 show_legend: bool = True,
                 interactive: bool = True,
                 **kwargs):
        """
        Initialize chart configuration.
        
        Args:
            width: Chart width in pixels
            height: Chart height in pixels
            theme: Chart theme
            show_volume: Whether to show volume subplot
            show_grid: Whether to show grid lines
            show_legend: Whether to show legend
            interactive: Whether chart is interactive
            **kwargs: Additional configuration options
        """
        self.width = width
        self.height = height
        self.theme = theme
        self.show_volume = show_volume
        self.show_grid = show_grid
        self.show_legend = show_legend
        self.interactive = interactive
        self.custom_config = kwargs
    
    def get_plotly_layout(self) -> Dict[str, Any]:
        """Get Plotly layout configuration."""
        layout = {
            'width': self.width,
            'height': self.height,
            'showlegend': self.show_legend,
            'hovermode': 'x unified' if self.interactive else False,
        }
        
        # Apply theme
        if self.theme == ChartTheme.TRADINGVIEW_DARK:
            layout.update({
                'template': 'plotly_dark',
                'paper_bgcolor': '#131722',
                'plot_bgcolor': '#131722',
                'font': {'color': '#d1d4dc'}
            })
        elif self.theme == ChartTheme.TRADINGVIEW_LIGHT:
            layout.update({
                'template': 'plotly_white',
                'paper_bgcolor': '#ffffff',
                'plot_bgcolor': '#ffffff'
            })
        
        # Grid settings
        if self.show_grid:
            layout.update({
                'xaxis': {'showgrid': True, 'gridcolor': '#2a2e39'},
                'yaxis': {'showgrid': True, 'gridcolor': '#2a2e39'}
            })
        
        # Merge custom configuration
        layout.update(self.custom_config)
        
        return layout


class ChartComponent(ABC):
    """Abstract base class for chart components."""
    
    def __init__(self, chart_type: ChartType, config: Optional[ChartConfig] = None):
        """
        Initialize chart component.
        
        Args:
            chart_type: Type of chart
            config: Chart configuration
        """
        self.chart_type = chart_type
        self.config = config or ChartConfig()
    
    @abstractmethod
    def render(self, data: ChartData) -> Any:
        """
        Render the chart component.
        
        Args:
            data: Chart data to render
            
        Returns:
            Plotly figure object
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: ChartData) -> bool:
        """
        Validate that data is compatible with this component.
        
        Args:
            data: Chart data to validate
            
        Returns:
            True if data is valid
        """
        pass
    
    def get_required_fields(self) -> List[str]:
        """Get list of required data fields."""
        return []
    
    def preprocess_data(self, data: ChartData) -> ChartData:
        """
        Preprocess data before rendering.
        
        Args:
            data: Original chart data
            
        Returns:
            Processed chart data
        """
        return data


class ChartFactory:
    """Factory for creating chart components."""
    
    _components: Dict[ChartType, type] = {}
    
    @classmethod
    def register_component(cls, chart_type: ChartType, component_class: type):
        """Register a chart component."""
        cls._components[chart_type] = component_class
    
    @classmethod
    def create_chart(cls, chart_type: ChartType, config: Optional[ChartConfig] = None) -> ChartComponent:
        """Create a chart component."""
        if chart_type not in cls._components:
            raise ValueError(f"Unknown chart type: {chart_type}")
        
        component_class = cls._components[chart_type]
        return component_class(chart_type, config)
    
    @classmethod
    def list_chart_types(cls) -> List[ChartType]:
        """List available chart types."""
        return list(cls._components.keys())


class VisualizationEngine:
    """Main visualization engine."""
    
    def __init__(self, default_config: Optional[ChartConfig] = None):
        """
        Initialize visualization engine.
        
        Args:
            default_config: Default chart configuration
        """
        self.default_config = default_config or ChartConfig()
        self.chart_factory = ChartFactory()
    
    def create_chart(self, 
                    chart_type: ChartType, 
                    data: ChartData,
                    config: Optional[ChartConfig] = None) -> Any:
        """
        Create and render a chart.
        
        Args:
            chart_type: Type of chart to create
            data: Chart data
            config: Chart configuration (uses default if None)
            
        Returns:
            Rendered chart (Plotly figure)
        """
        chart_config = config or self.default_config
        component = self.chart_factory.create_chart(chart_type, chart_config)
        
        if not component.validate_data(data):
            raise ValueError(f"Invalid data for chart type: {chart_type}")
        
        processed_data = component.preprocess_data(data)
        return component.render(processed_data)
    
    def create_dashboard_layout(self, 
                              charts: List[Dict[str, Any]],
                              layout_config: Optional[Dict] = None) -> List[Any]:
        """
        Create a dashboard layout with multiple charts.
        
        Args:
            charts: List of chart specifications
            layout_config: Layout configuration
            
        Returns:
            List of rendered charts for dashboard
        """
        rendered_charts = []
        
        for chart_spec in charts:
            chart_type = chart_spec['type']
            data = chart_spec['data']
            config = chart_spec.get('config', self.default_config)
            
            chart = self.create_chart(chart_type, data, config)
            rendered_charts.append(chart)
        
        return rendered_charts


class IndicatorOverlay:
    """Overlay for technical indicators."""
    
    def __init__(self, 
                 name: str, 
                 data: List[float],
                 color: str = 'blue',
                 line_width: int = 2,
                 opacity: float = 0.8):
        """
        Initialize indicator overlay.
        
        Args:
            name: Indicator name
            data: Indicator values
            color: Line color
            line_width: Line width
            opacity: Line opacity
        """
        self.name = name
        self.data = data
        self.color = color
        self.line_width = line_width
        self.opacity = opacity
    
    def to_plotly_trace(self, x_data: List[Any]) -> Any:
        """Convert to Plotly trace."""
        if go is None:
            raise ImportError("plotly is required for indicator overlays")
        
        return go.Scatter(
            x=x_data,
            y=self.data,
            mode='lines',
            name=self.name,
            line=dict(color=self.color, width=self.line_width),
            opacity=self.opacity
        )


class SignalOverlay:
    """Overlay for trading signals."""
    
    def __init__(self, 
                 buy_signals: List[int],
                 sell_signals: List[int],
                 prices: List[float]):
        """
        Initialize signal overlay.
        
        Args:
            buy_signals: List of buy signal indices
            sell_signals: List of sell signal indices
            prices: Price data for signal positioning
        """
        self.buy_signals = buy_signals
        self.sell_signals = sell_signals
        self.prices = prices
    
    def to_plotly_traces(self, x_data: List[Any]) -> List[Any]:
        """Convert to Plotly traces."""
        if go is None:
            raise ImportError("plotly is required for signal overlays")
        
        traces = []
        
        # Buy signals
        if self.buy_signals:
            buy_x = [x_data[i] for i in self.buy_signals if i < len(x_data)]
            buy_y = [self.prices[i] for i in self.buy_signals if i < len(self.prices)]
            
            traces.append(go.Scatter(
                x=buy_x,
                y=buy_y,
                mode='markers',
                name='Buy Signals',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green',
                    line=dict(color='darkgreen', width=2)
                )
            ))
        
        # Sell signals
        if self.sell_signals:
            sell_x = [x_data[i] for i in self.sell_signals if i < len(x_data)]
            sell_y = [self.prices[i] for i in self.sell_signals if i < len(self.prices)]
            
            traces.append(go.Scatter(
                x=sell_x,
                y=sell_y,
                mode='markers',
                name='Sell Signals',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(color='darkred', width=2)
                )
            ))
        
        return traces