"""
Data caching and management.
"""

import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Union
from pathlib import Path
import pandas as pd

from configs.settings import get_settings


class CacheManager:
    """Manage data caching with TTL and versioning."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.settings = get_settings()
        self.cache_dir = Path(cache_dir or "data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = self.settings.cache_ttl_seconds
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key hash."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str, extension: str = "pkl") -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.{extension}"
    
    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get cache metadata file path."""
        return self.cache_dir / f"{cache_key}.meta"
    
    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        if 'expires_at' not in metadata:
            return True
        
        expires_at = datetime.fromisoformat(metadata['expires_at'])
        return datetime.now() > expires_at
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Set a cache entry.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tags: Optional tags for cache invalidation
        """
        cache_key = self._get_cache_key(key)
        ttl = ttl or self.default_ttl
        
        # Prepare metadata
        metadata = {
            'key': key,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(seconds=ttl)).isoformat(),
            'ttl': ttl,
            'tags': tags or [],
            'size_bytes': 0,
            'type': type(value).__name__
        }
        
        # Choose storage format based on data type
        if isinstance(value, pd.DataFrame):
            # Store DataFrames as parquet for efficiency
            cache_path = self._get_cache_path(cache_key, "parquet")
            value.to_parquet(cache_path)
            metadata['format'] = 'parquet'
        elif isinstance(value, (dict, list)) and all(
            isinstance(item, (str, int, float, bool, type(None))) 
            for item in (value.values() if isinstance(value, dict) else value)
        ):
            # Store simple JSON-serializable data as JSON
            cache_path = self._get_cache_path(cache_key, "json")
            with open(cache_path, 'w') as f:
                json.dump(value, f)
            metadata['format'] = 'json'
        else:
            # Store complex objects as pickle
            cache_path = self._get_cache_path(cache_key, "pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            metadata['format'] = 'pickle'
        
        # Store metadata
        metadata['size_bytes'] = cache_path.stat().st_size
        metadata_path = self._get_metadata_path(cache_key)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a cache entry."""
        cache_key = self._get_cache_key(key)
        metadata_path = self._get_metadata_path(cache_key)
        
        if not metadata_path.exists():
            return None
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check if expired
        if self._is_expired(metadata):
            self.delete(key)
            return None
        
        # Load data based on format
        cache_format = metadata.get('format', 'pickle')
        
        if cache_format == 'parquet':
            cache_path = self._get_cache_path(cache_key, "parquet")
            return pd.read_parquet(cache_path)
        elif cache_format == 'json':
            cache_path = self._get_cache_path(cache_key, "json")
            with open(cache_path, 'r') as f:
                return json.load(f)
        else:  # pickle
            cache_path = self._get_cache_path(cache_key, "pkl")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    
    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        cache_key = self._get_cache_key(key)
        deleted = False
        
        # Delete all possible file formats
        for extension in ['pkl', 'json', 'parquet']:
            cache_path = self._get_cache_path(cache_key, extension)
            if cache_path.exists():
                cache_path.unlink()
                deleted = True
        
        # Delete metadata
        metadata_path = self._get_metadata_path(cache_key)
        if metadata_path.exists():
            metadata_path.unlink()
            deleted = True
        
        return deleted
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern."""
        cleared = 0
        
        for metadata_file in self.cache_dir.glob("*.meta"):
            cache_key = metadata_file.stem
            
            # Load metadata to get original key
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            original_key = metadata.get('key', '')
            
            # Check pattern match
            if pattern is None or pattern in original_key:
                if self.delete(original_key):
                    cleared += 1
        
        return cleared
    
    def clear_by_tag(self, tag: str) -> int:
        """Clear cache entries with specific tag."""
        cleared = 0
        
        for metadata_file in self.cache_dir.glob("*.meta"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if tag in metadata.get('tags', []):
                original_key = metadata.get('key', '')
                if self.delete(original_key):
                    cleared += 1
        
        return cleared
    
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        cleaned = 0
        
        for metadata_file in self.cache_dir.glob("*.meta"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if self._is_expired(metadata):
                original_key = metadata.get('key', '')
                if self.delete(original_key):
                    cleaned += 1
        
        return cleaned
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = 0
        total_size = 0
        expired_entries = 0
        formats = {}
        
        for metadata_file in self.cache_dir.glob("*.meta"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            total_entries += 1
            total_size += metadata.get('size_bytes', 0)
            
            if self._is_expired(metadata):
                expired_entries += 1
            
            format_type = metadata.get('format', 'unknown')
            formats[format_type] = formats.get(format_type, 0) + 1
        
        return {
            'total_entries': total_entries,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'expired_entries': expired_entries,
            'active_entries': total_entries - expired_entries,
            'formats': formats,
            'cache_dir': str(self.cache_dir)
        }


class DataVersionManager:
    """Manage data versioning and archival."""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.settings = get_settings()
        self.data_dir = Path(data_dir or "data")
        self.archives_dir = self.data_dir / "archives"
        self.archives_dir.mkdir(parents=True, exist_ok=True)
    
    def archive_data(self, data: pd.DataFrame, name: str, version: Optional[str] = None) -> str:
        """Archive data with versioning."""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{name}_v{version}.parquet"
        filepath = self.archives_dir / filename
        
        # Add metadata
        metadata = {
            'name': name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'rows': len(data),
            'columns': list(data.columns),
            'size_bytes': 0
        }
        
        # Save data
        data.to_parquet(filepath)
        metadata['size_bytes'] = filepath.stat().st_size
        
        # Save metadata
        metadata_path = filepath.with_suffix('.meta')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return version
    
    def load_data(self, name: str, version: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load archived data."""
        if version is None:
            # Get latest version
            pattern = f"{name}_v*.parquet"
            files = list(self.archives_dir.glob(pattern))
            if not files:
                return None
            
            # Sort by modification time, get latest
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
        else:
            filename = f"{name}_v{version}.parquet"
            latest_file = self.archives_dir / filename
            if not latest_file.exists():
                return None
        
        return pd.read_parquet(latest_file)
    
    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a dataset."""
        pattern = f"{name}_v*.meta"
        versions = []
        
        for meta_file in self.archives_dir.glob(pattern):
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            versions.append(metadata)
        
        # Sort by creation time
        versions.sort(key=lambda v: v['created_at'], reverse=True)
        return versions
    
    def cleanup_old_versions(self, name: str, keep_versions: int = 5) -> int:
        """Clean up old versions, keeping only the most recent."""
        versions = self.list_versions(name)
        
        if len(versions) <= keep_versions:
            return 0
        
        # Delete old versions
        versions_to_delete = versions[keep_versions:]
        deleted = 0
        
        for version_info in versions_to_delete:
            version = version_info['version']
            
            # Delete data file
            data_file = self.archives_dir / f"{name}_v{version}.parquet"
            if data_file.exists():
                data_file.unlink()
                deleted += 1
            
            # Delete metadata file
            meta_file = self.archives_dir / f"{name}_v{version}.meta"
            if meta_file.exists():
                meta_file.unlink()
        
        return deleted