"""
Dependency injection container.
"""

from typing import TypeVar, Type, Dict, Any, Callable
from functools import lru_cache
import inspect

from configs.settings import get_settings


T = TypeVar('T')


class Container:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        
        # Register default services
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default services."""
        # Settings
        self.register_singleton("settings", get_settings)
        
        # TODO: Register other default services like database, cache, etc.
    
    def register(self, name: str, service: Any) -> None:
        """Register a service instance."""
        self._services[name] = service
    
    def register_factory(self, name: str, factory: Callable) -> None:
        """Register a factory function for a service."""
        self._factories[name] = factory
    
    def register_singleton(self, name: str, factory: Callable) -> None:
        """Register a singleton service with factory."""
        self._factories[name] = factory
        # Mark as singleton by storing in separate dict
        self._singletons[name] = None
    
    def get(self, name: str) -> Any:
        """Get a service by name."""
        # Check if it's a direct service
        if name in self._services:
            return self._services[name]
        
        # Check if it's a singleton
        if name in self._singletons:
            if self._singletons[name] is None:
                # Create singleton instance
                factory = self._factories[name]
                self._singletons[name] = self._create_instance(factory)
            return self._singletons[name]
        
        # Check if it's a factory
        if name in self._factories:
            factory = self._factories[name]
            return self._create_instance(factory)
        
        raise ValueError(f"Service '{name}' not found")
    
    def get_by_type(self, service_type: Type[T]) -> T:
        """Get a service by its type."""
        type_name = service_type.__name__.lower()
        return self.get(type_name)
    
    def _create_instance(self, factory: Callable) -> Any:
        """Create an instance using dependency injection."""
        # Get factory signature
        sig = inspect.signature(factory)
        kwargs = {}
        
        # Resolve dependencies
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                # Try to resolve by type
                try:
                    dependency = self.get_by_type(param.annotation)
                    kwargs[param_name] = dependency
                except ValueError:
                    # Try to resolve by name
                    try:
                        dependency = self.get(param_name)
                        kwargs[param_name] = dependency
                    except ValueError:
                        # Use default value if available
                        if param.default != inspect.Parameter.empty:
                            kwargs[param_name] = param.default
                        else:
                            raise ValueError(
                                f"Cannot resolve dependency '{param_name}' "
                                f"for factory {factory.__name__}"
                            )
        
        return factory(**kwargs)
    
    def autowire(self, cls: Type[T]) -> T:
        """Create an instance of a class with automatic dependency injection."""
        return self._create_instance(cls)


# Global container instance
_container = Container()


def get_container() -> Container:
    """Get the global container instance."""
    return _container


def inject(name: str):
    """Decorator for dependency injection."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Inject the dependency
            dependency = _container.get(name)
            return func(dependency, *args, **kwargs)
        return wrapper
    return decorator


def singleton(name: str = None):
    """Decorator to register a class as a singleton service."""
    def decorator(cls):
        service_name = name or cls.__name__.lower()
        _container.register_singleton(service_name, cls)
        return cls
    return decorator


def service(name: str = None):
    """Decorator to register a class as a service factory."""
    def decorator(cls):
        service_name = name or cls.__name__.lower()
        _container.register_factory(service_name, cls)
        return cls
    return decorator