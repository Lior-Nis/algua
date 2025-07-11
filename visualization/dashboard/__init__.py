"""
Dashboard package.
"""

try:
    from .streamlit_dashboard import AlguaDashboard
    
    __all__ = ['AlguaDashboard']
except ImportError:
    # streamlit not available
    __all__ = []