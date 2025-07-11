"""
Visualization module for Algua trading platform.
"""

from .interfaces.chart_interfaces import (
    ChartType, ChartTheme, ChartData, ChartConfig, 
    ChartComponent, ChartFactory, VisualizationEngine,
    IndicatorOverlay, SignalOverlay
)

try:
    from .charts.tradingview_charts import (
        CandlestickChart, LineChart, PortfolioChart, VolumeChart
    )
except ImportError:
    # plotly not available
    pass

__all__ = [
    'ChartType', 'ChartTheme', 'ChartData', 'ChartConfig',
    'ChartComponent', 'ChartFactory', 'VisualizationEngine',
    'IndicatorOverlay', 'SignalOverlay'
]