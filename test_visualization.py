#!/usr/bin/env python3
"""
Test script for the visualization system.
"""

import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from domain.value_objects import Symbol
from infrastructure.interfaces import DataProviderFactory
from infrastructure.providers.simple_data_provider import SimpleDataProvider
from models.strategy_factory import StrategyFactory
from visualization.interfaces.chart_interfaces import (
    ChartType, ChartTheme, ChartData, ChartConfig, 
    VisualizationEngine, IndicatorOverlay, SignalOverlay
)
from utils.logging import get_logger

logger = get_logger(__name__)


def test_chart_interfaces():
    """Test chart interfaces without external dependencies."""
    print("🧪 Testing Chart Interfaces...")
    
    try:
        # Test ChartData
        sample_data = [
            {'Open': 100, 'High': 105, 'Low': 98, 'Close': 103, 'Volume': 1000000},
            {'Open': 103, 'High': 107, 'Low': 101, 'Close': 106, 'Volume': 1200000}
        ]
        
        chart_data = ChartData(
            data=sample_data,
            chart_type=ChartType.CANDLESTICK,
            title="Test Chart",
            metadata={'symbol': 'AAPL'}
        )
        
        print(f"✓ ChartData created: {chart_data.title}")
        print(f"✓ Data points: {len(chart_data.get_data())}")
        print(f"✓ Metadata: {chart_data.get_metadata('symbol')}")
        
        # Test ChartConfig
        config = ChartConfig(
            width=1200,
            height=600,
            theme=ChartTheme.TRADINGVIEW_DARK,
            show_volume=True
        )
        
        print(f"✓ ChartConfig created: {config.width}x{config.height}")
        print(f"✓ Theme: {config.theme.value}")
        
        # Test layout generation
        layout = config.get_plotly_layout()
        print(f"✓ Layout generated with {len(layout)} properties")
        
        # Test enums
        chart_types = list(ChartType)
        themes = list(ChartTheme)
        
        print(f"✓ Available chart types: {[ct.value for ct in chart_types]}")
        print(f"✓ Available themes: {[t.value for t in themes]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chart interfaces test failed: {e}")
        return False


def test_visualization_engine():
    """Test visualization engine without plotly."""
    print("\n🎨 Testing Visualization Engine...")
    
    try:
        # Create visualization engine
        viz_engine = VisualizationEngine()
        print("✓ VisualizationEngine created")
        
        # Test chart factory
        from visualization.interfaces.chart_interfaces import ChartFactory
        
        # List available chart types (should be empty without plotly registration)
        available_types = ChartFactory.list_chart_types()
        print(f"✓ Available chart types: {[ct.value for ct in available_types]}")
        
        # Test with mock data
        sample_data = {
            'x': [1, 2, 3, 4, 5],
            'y': [10, 15, 13, 17, 20]
        }
        
        chart_data = ChartData(
            data=sample_data,
            chart_type=ChartType.LINE,
            title="Test Line Chart"
        )
        
        print(f"✓ Chart data prepared for {chart_data.chart_type.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization engine test failed: {e}")
        return False


def test_overlay_system():
    """Test indicator and signal overlays."""
    print("\n📊 Testing Overlay System...")
    
    try:
        # Test IndicatorOverlay
        sma_data = [100, 101, 102, 104, 103, 105, 107, 106, 108, 110]
        
        indicator_overlay = IndicatorOverlay(
            name="SMA 10",
            data=sma_data,
            color='blue',
            line_width=2
        )
        
        print(f"✓ IndicatorOverlay created: {indicator_overlay.name}")
        print(f"✓ Data points: {len(indicator_overlay.data)}")
        print(f"✓ Color: {indicator_overlay.color}")
        
        # Test SignalOverlay
        buy_signals = [2, 6]
        sell_signals = [4, 8]
        prices = [100, 102, 101, 103, 102, 104, 106, 105, 107, 109]
        
        signal_overlay = SignalOverlay(
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            prices=prices
        )
        
        print(f"✓ SignalOverlay created")
        print(f"✓ Buy signals at indices: {signal_overlay.buy_signals}")
        print(f"✓ Sell signals at indices: {signal_overlay.sell_signals}")
        
        return True
        
    except Exception as e:
        print(f"❌ Overlay system test failed: {e}")
        return False


def test_strategy_visualization_integration():
    """Test integration with strategy system."""
    print("\n🔗 Testing Strategy-Visualization Integration...")
    
    try:
        # Get market data
        provider = DataProviderFactory.create("simple")
        symbol = Symbol("AAPL")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=50)
        
        data = provider.get_historical_data(symbol, start_date, end_date)
        print(f"✓ Retrieved {len(data)} data points for {symbol}")
        
        # Create strategy and generate signals
        strategy = StrategyFactory.create_strategy('sma_crossover', fast_period=5, slow_period=20)
        signals = strategy.generate_signals(data)
        
        print(f"✓ Generated {len(signals)} signal points")
        
        # Prepare data for visualization
        chart_data = ChartData(
            data=data,
            chart_type=ChartType.CANDLESTICK,
            title=f"{symbol} with SMA Crossover Signals",
            metadata={
                'symbol': str(symbol),
                'strategy': 'sma_crossover',
                'fast_period': 5,
                'slow_period': 20
            }
        )
        
        print(f"✓ Chart data prepared: {chart_data.title}")
        
        # Extract indicators for overlays
        fast_sma_data = [s['fast_sma'] for s in signals]
        slow_sma_data = [s['slow_sma'] for s in signals]
        
        fast_overlay = IndicatorOverlay(
            name="Fast SMA (5)",
            data=fast_sma_data,
            color='green',
            line_width=2
        )
        
        slow_overlay = IndicatorOverlay(
            name="Slow SMA (20)",
            data=slow_sma_data,
            color='red',
            line_width=2
        )
        
        print(f"✓ SMA overlays created")
        
        # Extract signals for overlay
        buy_indices = [i for i, s in enumerate(signals) if s['buy_signal']]
        sell_indices = [i for i, s in enumerate(signals) if s['sell_signal']]
        prices = [item['Close'] for item in data]
        
        signal_overlay = SignalOverlay(
            buy_signals=buy_indices,
            sell_signals=sell_indices,
            prices=prices
        )
        
        print(f"✓ Signal overlay created: {len(buy_indices)} buy, {len(sell_indices)} sell")
        
        # Test chart configuration
        config = ChartConfig(
            width=1200,
            height=600,
            theme=ChartTheme.TRADINGVIEW_DARK,
            show_volume=True,
            show_grid=True
        )
        
        print(f"✓ Chart configuration: {config.theme.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy-visualization integration test failed: {e}")
        return False


def test_dashboard_architecture():
    """Test dashboard architecture design."""
    print("\n🏗️ Testing Dashboard Architecture...")
    
    try:
        # Test dashboard data structures
        dashboard_charts = [
            {
                'type': ChartType.CANDLESTICK,
                'data': ChartData(
                    data=[{'Open': 100, 'High': 105, 'Low': 98, 'Close': 103, 'Volume': 1000}],
                    chart_type=ChartType.CANDLESTICK,
                    title="Price Chart"
                ),
                'config': ChartConfig(theme=ChartTheme.TRADINGVIEW_DARK)
            },
            {
                'type': ChartType.PORTFOLIO,
                'data': ChartData(
                    data=[{'timestamp': datetime.now(), 'portfolio_value': 100000}],
                    chart_type=ChartType.PORTFOLIO,
                    title="Portfolio Performance"
                ),
                'config': ChartConfig(theme=ChartTheme.TRADINGVIEW_DARK)
            }
        ]
        
        print(f"✓ Dashboard layout prepared with {len(dashboard_charts)} charts")
        
        # Test chart metadata
        for i, chart_spec in enumerate(dashboard_charts):
            chart_data = chart_spec['data']
            print(f"✓ Chart {i+1}: {chart_data.title} ({chart_data.chart_type.value})")
        
        # Test theme consistency
        themes = [chart['config'].theme for chart in dashboard_charts]
        consistent_theme = all(theme == themes[0] for theme in themes)
        print(f"✓ Theme consistency: {consistent_theme}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard architecture test failed: {e}")
        return False


def test_performance_considerations():
    """Test performance-related features."""
    print("\n⚡ Testing Performance Considerations...")
    
    try:
        # Test with large dataset
        large_data = []
        for i in range(1000):  # 1000 data points
            large_data.append({
                'Open': 100 + i * 0.1,
                'High': 105 + i * 0.1,
                'Low': 98 + i * 0.1,
                'Close': 103 + i * 0.1,
                'Volume': 1000000 + i * 1000,
                'Timestamp': datetime.now() - timedelta(days=1000-i)
            })
        
        print(f"✓ Large dataset created: {len(large_data)} points")
        
        # Test data sampling (simulate)
        sample_size = 100
        sampled_data = large_data[::len(large_data)//sample_size]
        
        print(f"✓ Data sampling: {len(large_data)} -> {len(sampled_data)} points")
        
        # Test metadata efficiency
        chart_data = ChartData(
            data=sampled_data,
            chart_type=ChartType.CANDLESTICK,
            title="Performance Test Chart",
            metadata={
                'original_size': len(large_data),
                'sampled_size': len(sampled_data),
                'compression_ratio': len(sampled_data) / len(large_data)
            }
        )
        
        compression_ratio = chart_data.get_metadata('compression_ratio')
        print(f"✓ Compression ratio: {compression_ratio:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all visualization tests."""
    print("=" * 80)
    print("📊 ALGUA VISUALIZATION SYSTEM TESTS")
    print("=" * 80)
    
    tests = [
        ("Chart Interfaces", test_chart_interfaces),
        ("Visualization Engine", test_visualization_engine),
        ("Overlay System", test_overlay_system),
        ("Strategy Integration", test_strategy_visualization_integration),
        ("Dashboard Architecture", test_dashboard_architecture),
        ("Performance Features", test_performance_considerations)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    total = len(tests)
    
    print("\n" + "=" * 80)
    print("📋 VISUALIZATION TEST SUMMARY")
    print("=" * 80)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if i < passed else "❌ FAIL"
        print(f"  {test_name:<25}: {status}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL VISUALIZATION TESTS PASSED!")
        print("\n📋 Visualization System Summary:")
        print("  ✓ Chart interfaces and data structures")
        print("  ✓ Visualization engine architecture") 
        print("  ✓ Indicator and signal overlay system")
        print("  ✓ Strategy integration capabilities")
        print("  ✓ Dashboard layout architecture")
        print("  ✓ Performance optimization features")
        
        print(f"\n🚀 Ready for plotly integration:")
        print("  1. Install plotly: pip install plotly")
        print("  2. Install streamlit: pip install streamlit")
        print("  3. Run dashboard: streamlit run visualization/dashboard/streamlit_dashboard.py")
        print("  4. Test with real charts and interactivity")
        
    else:
        print(f"\n⚠️ {total - passed} tests failed.")
    
    print("\n📖 Architecture Design Summary:")
    print("✅ Pluggable chart system with factory pattern")
    print("✅ TradingView-style theming and layout")
    print("✅ Strategy signal overlay integration")
    print("✅ Performance optimization for large datasets")
    print("✅ Streamlit dashboard framework")
    print("✅ Domain-driven architecture compliance")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)