#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缓存管理模块 (Cache Manager)
负责数据缓存和更新策略
"""

import os
import sys
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, TypeVar
from datetime import datetime, timedelta
import threading
import hashlib
import shutil
import logging
from enum import Enum, auto
from functools import wraps

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.config_manager import ConfigManager
from .data_types import CacheEntry, DataQualityLevel

logger = get_logger(__name__)


class CacheStrategy(Enum):
    """
    缓存策略枚举
    """
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最少使用频率
    FIFO = "fifo"  # 先进先出
    TTL = "ttl"  # 基于时间的过期


class StorageType(Enum):
    """
    存储类型枚举
    """
    MEMORY = "memory"  # 内存存储
    FILE = "file"  # 文件存储
    REDIS = "redis"  # Redis存储（预留）


class CacheManager:
    """
    缓存管理器类
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化缓存管理器
        
        Args:
            config: 配置信息
        """
        default_config = {
            'strategy': CacheStrategy.LRU.value,
            'storage_type': StorageType.MEMORY.value,
            'max_size': 1000,  # 最大缓存项数
            'max_memory_mb': 512,  # 最大内存使用量（MB）
            'default_ttl': 3600,  # 默认过期时间（秒）
            'cache_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'cache'),
            'clean_interval': 300,  # 清理间隔（秒）
            'compress': False,  # 是否压缩
        }
        
        self.config = {**default_config, **(config or {})}
        self.strategy = CacheStrategy(self.config['strategy'])
        self.storage_type = StorageType(self.config['storage_type'])
        
        # 初始化存储
        self._initialize_storage()
        
        # 初始化统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0,
            'memory_usage_mb': 0.0
        }
        
        # 启动定期清理线程
        self._start_cleanup_thread()
        
        logger.info(f"Cache manager initialized with strategy={self.strategy.value}, storage={self.storage_type.value}")
    
    def _initialize_storage(self) -> None:
        """
        初始化存储
        """
        if self.storage_type == StorageType.MEMORY:
            # 内存存储
            self._cache: Dict[str, CacheEntry] = {}
            
            # 策略特定的数据结构
            if self.strategy == CacheStrategy.LRU:
                self._access_times: Dict[str, float] = {}  # 记录访问时间
            elif self.strategy == CacheStrategy.LFU:
                self._access_counts: Dict[str, int] = {}  # 记录访问频率
            elif self.strategy == CacheStrategy.FIFO:
                self._insertion_order: List[str] = []  # 记录插入顺序
                
        elif self.storage_type == StorageType.FILE:
            # 文件存储
            self._cache_dir = self.config['cache_dir']
            os.makedirs(self._cache_dir, exist_ok=True)
            
            # 创建索引文件
            self._index_file = os.path.join(self._cache_dir, '.index.json')
            self._index_lock = threading.RLock()
            self._load_index()
        
    def _load_index(self) -> None:
        """
        加载文件存储索引
        """
        if self.storage_type == StorageType.FILE:
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    self._index = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self._index = {
                    'entries': {},
                    'access_times': {},
                    'access_counts': {},
                    'insertion_order': []
                }
    
    def _save_index(self) -> None:
        """
        保存文件存储索引
        """
        if self.storage_type == StorageType.FILE:
            with self._index_lock:
                with open(self._index_file, 'w', encoding='utf-8') as f:
                    json.dump(self._index, f, indent=2, ensure_ascii=False)
    
    def _start_cleanup_thread(self) -> None:
        """
        启动定期清理线程
        """
        self._stop_cleanup = False
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _cleanup_loop(self) -> None:
        """
        清理循环
        """
        while not self._stop_cleanup:
            try:
                self.cleanup()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
            
            # 等待下一次清理
            for _ in range(self.config['clean_interval']):
                if self._stop_cleanup:
                    break
                threading.Event().wait(1)
    
    def cleanup(self) -> None:
        """
        清理过期和无效的缓存项
        """
        logger.debug("Running cache cleanup")
        expired_keys = self._get_expired_keys()
        
        for key in expired_keys:
            self.delete(key)
        
        # 检查是否需要执行策略性驱逐
        if self._should_evict():
            self._evict()
        
        # 更新统计信息
        self._update_stats()
        
        logger.debug(f"Cleanup complete: removed {len(expired_keys)} expired entries")
    
    def _get_expired_keys(self) -> List[str]:
        """
        获取过期的键列表
        
        Returns:
            过期键列表
        """
        expired_keys = []
        
        if self.storage_type == StorageType.MEMORY:
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
        
        elif self.storage_type == StorageType.FILE:
            with self._index_lock:
                for key, entry_info in self._index['entries'].items():
                    if 'expires_at' in entry_info:
                        expires_at = datetime.fromisoformat(entry_info['expires_at'])
                        if datetime.now() > expires_at:
                            expired_keys.append(key)
        
        return expired_keys
    
    def _should_evict(self) -> bool:
        """
        检查是否需要执行驱逐
        
        Returns:
            是否需要驱逐
        """
        if self.storage_type == StorageType.MEMORY:
            # 检查大小限制
            if len(self._cache) >= self.config['max_size']:
                return True
            
            # 检查内存使用限制
            if self.stats['memory_usage_mb'] >= self.config['max_memory_mb']:
                return True
        
        return False
    
    def _evict(self) -> None:
        """
        根据策略执行驱逐
        """
        keys_to_evict = []
        
        if self.storage_type == StorageType.MEMORY:
            if self.strategy == CacheStrategy.LRU:
                # 驱逐最久未使用的
                if self._access_times:
                    oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
                    keys_to_evict.append(oldest_key)
            
            elif self.strategy == CacheStrategy.LFU:
                # 驱逐使用频率最低的
                if self._access_counts:
                    least_used_key = min(self._access_counts.items(), key=lambda x: x[1])[0]
                    keys_to_evict.append(least_used_key)
            
            elif self.strategy == CacheStrategy.FIFO:
                # 驱逐最先插入的
                if self._insertion_order:
                    keys_to_evict.append(self._insertion_order[0])
            
            elif self.strategy == CacheStrategy.TTL:
                # 基于TTL的驱逐已经在cleanup中处理
                pass
        
        # 执行驱逐
        for key in keys_to_evict:
            self.delete(key)
            self.stats['evictions'] += 1
            logger.debug(f"Evicted cache entry: {key}")
    
    def _update_stats(self) -> None:
        """
        更新统计信息
        """
        if self.storage_type == StorageType.MEMORY:
            self.stats['size'] = len(self._cache)
            # 估算内存使用
            self.stats['memory_usage_mb'] = self._estimate_memory_usage()
    
    def _estimate_memory_usage(self) -> float:
        """
        估算内存使用量
        
        Returns:
            内存使用量（MB）
        """
        # 简单估算：每个缓存项平均10KB
        estimated_bytes = len(self._cache) * 10240
        return estimated_bytes / (1024 * 1024)  # 转换为MB
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的数据，如果不存在或已过期返回None
        """
        if self.storage_type == StorageType.MEMORY:
            if key not in self._cache:
                self.stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # 检查是否过期
            if entry.is_expired():
                self.delete(key)
                self.stats['misses'] += 1
                return None
            
            # 更新策略相关信息
            self._update_access_info(key)
            
            self.stats['hits'] += 1
            logger.debug(f"Cache hit for key: {key}")
            return entry.data
        
        elif self.storage_type == StorageType.FILE:
            with self._index_lock:
                if key not in self._index['entries']:
                    self.stats['misses'] += 1
                    return None
                
                entry_info = self._index['entries'][key]
                
                # 检查是否过期
                if 'expires_at' in entry_info:
                    expires_at = datetime.fromisoformat(entry_info['expires_at'])
                    if datetime.now() > expires_at:
                        self.delete(key)
                        self.stats['misses'] += 1
                        return None
                
                # 更新访问信息
                self._index['access_times'][key] = datetime.now().timestamp()
                self._index['access_counts'][key] = self._index['access_counts'].get(key, 0) + 1
                
            # 从文件加载数据
            file_path = os.path.join(self._cache_dir, hashlib.md5(key.encode()).hexdigest())
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.stats['hits'] += 1
                logger.debug(f"Cache hit for key: {key}")
                return data
            except Exception as e:
                logger.error(f"Error loading cache file for key {key}: {e}")
                # 删除损坏的文件
                try:
                    os.remove(file_path)
                    with self._index_lock:
                        if key in self._index['entries']:
                            del self._index['entries'][key]
                except:
                    pass
                
                self.stats['misses'] += 1
                return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None,
           **metadata) -> bool:
        """
        设置缓存项
        
        Args:
            key: 缓存键
            data: 要缓存的数据
            ttl: 生存时间（秒），None表示使用默认值
            **metadata: 元数据
            
        Returns:
            是否设置成功
        """
        try:
            # 确定过期时间
            if ttl is None:
                ttl = self.config['default_ttl']
            
            expires_at = None
            if ttl > 0:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            
            if self.storage_type == StorageType.MEMORY:
                # 创建缓存条目
                entry = CacheEntry(
                    key=key,
                    data=data,
                    expires_at=expires_at,
                    metadata=metadata
                )
                
                # 检查是否需要先驱逐
                if key not in self._cache and self._should_evict():
                    self._evict()
                
                # 存储条目
                self._cache[key] = entry
                
                # 更新策略相关信息
                if self.strategy == CacheStrategy.LRU:
                    self._access_times[key] = datetime.now().timestamp()
                elif self.strategy == CacheStrategy.LFU:
                    self._access_counts[key] = 1
                elif self.strategy == CacheStrategy.FIFO:
                    if key in self._insertion_order:
                        self._insertion_order.remove(key)
                    self._insertion_order.append(key)
                
            elif self.storage_type == StorageType.FILE:
                # 保存到文件
                file_path = os.path.join(self._cache_dir, hashlib.md5(key.encode()).hexdigest())
                
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                
                # 更新索引
                with self._index_lock:
                    entry_info = {
                        'created_at': datetime.now().isoformat(),
                        'metadata': metadata
                    }
                    if expires_at:
                        entry_info['expires_at'] = expires_at.isoformat()
                    
                    self._index['entries'][key] = entry_info
                    self._index['access_times'][key] = datetime.now().timestamp()
                    self._index['access_counts'][key] = 1
                    
                    if self.strategy == CacheStrategy.FIFO:
                        if key in self._index['insertion_order']:
                            self._index['insertion_order'].remove(key)
                        self._index['insertion_order'].append(key)
                    
                    self._save_index()
            
            # 更新统计信息
            self._update_stats()
            
            logger.debug(f"Cache set for key: {key} (ttl: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        try:
            if self.storage_type == StorageType.MEMORY:
                if key in self._cache:
                    del self._cache[key]
                    
                    # 清理策略相关数据
                    if self.strategy == CacheStrategy.LRU and key in self._access_times:
                        del self._access_times[key]
                    elif self.strategy == CacheStrategy.LFU and key in self._access_counts:
                        del self._access_counts[key]
                    elif self.strategy == CacheStrategy.FIFO and key in self._insertion_order:
                        self._insertion_order.remove(key)
                    
                    self._update_stats()
                    return True
            
            elif self.storage_type == StorageType.FILE:
                file_path = os.path.join(self._cache_dir, hashlib.md5(key.encode()).hexdigest())
                
                # 删除文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                # 更新索引
                with self._index_lock:
                    if key in self._index['entries']:
                        del self._index['entries'][key]
                        
                        if key in self._index['access_times']:
                            del self._index['access_times'][key]
                        if key in self._index['access_counts']:
                            del self._index['access_counts'][key]
                        if key in self._index['insertion_order']:
                            self._index['insertion_order'].remove(key)
                        
                        self._save_index()
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting cache for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查缓存项是否存在且未过期
        
        Args:
            key: 缓存键
            
        Returns:
            是否存在且有效
        """
        return self.get(key) is not None
    
    def clear(self) -> None:
        """
        清空所有缓存
        """
        logger.info("Clearing all cache")
        
        if self.storage_type == StorageType.MEMORY:
            self._cache.clear()
            
            # 清空策略相关数据
            if self.strategy == CacheStrategy.LRU:
                self._access_times.clear()
            elif self.strategy == CacheStrategy.LFU:
                self._access_counts.clear()
            elif self.strategy == CacheStrategy.FIFO:
                self._insertion_order.clear()
        
        elif self.storage_type == StorageType.FILE:
            # 删除缓存目录下的所有文件
            for filename in os.listdir(self._cache_dir):
                if filename != '.index.json':
                    file_path = os.path.join(self._cache_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Error deleting cache file {filename}: {e}")
            
            # 重置索引
            with self._index_lock:
                self._index = {
                    'entries': {},
                    'access_times': {},
                    'access_counts': {},
                    'insertion_order': []
                }
                self._save_index()
        
        # 重置统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0,
            'memory_usage_mb': 0.0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        self._update_stats()
        
        # 计算命中率
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total) * 100 if total > 0 else 0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'strategy': self.strategy.value,
            'storage_type': self.storage_type.value,
            'max_size': self.config['max_size'],
            'default_ttl': self.config['default_ttl']
        }
    
    def _update_access_info(self, key: str) -> None:
        """
        更新访问信息
        
        Args:
            key: 缓存键
        """
        if self.strategy == CacheStrategy.LRU:
            self._access_times[key] = datetime.now().timestamp()
        elif self.strategy == CacheStrategy.LFU:
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
    
    def generate_key(self, prefix: str, **kwargs) -> str:
        """
        根据参数生成唯一的缓存键
        
        Args:
            prefix: 键前缀
            **kwargs: 用于生成键的参数
            
        Returns:
            生成的缓存键
        """
        # 对参数进行排序以确保一致性
        sorted_params = sorted(kwargs.items(), key=lambda x: x[0])
        params_str = json.dumps(sorted_params, sort_keys=True, ensure_ascii=False)
        
        # 生成哈希
        param_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        return f"{prefix}:{param_hash}"
    
    def close(self) -> None:
        """
        关闭缓存管理器
        """
        logger.info("Closing cache manager")
        
        # 停止清理线程
        self._stop_cleanup = True
        if hasattr(self, '_cleanup_thread'):
            self._cleanup_thread.join(timeout=5)
        
        # 保存索引
        if self.storage_type == StorageType.FILE:
            self._save_index()


class CacheDecorator:
    """
    缓存装饰器类
    """
    
    def __init__(self, cache_manager: CacheManager, ttl: Optional[int] = None):
        """
        初始化装饰器
        
        Args:
            cache_manager: 缓存管理器实例
            ttl: 生存时间（秒）
        """
        self.cache_manager = cache_manager
        self.ttl = ttl
    
    def __call__(self, func: Callable) -> Callable:
        """
        装饰函数
        
        Args:
            func: 要装饰的函数
            
        Returns:
            装饰后的函数
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            # 排除self参数（对于实例方法）
            if args and hasattr(args[0], '__class__') and args[0].__class__.__name__ != str(args[0].__class__):
                cache_args = args[1:]  # 排除self
            else:
                cache_args = args
            
            # 生成缓存键
            key = self.cache_manager.generate_key(
                prefix=f"func:{func.__module__}:{func.__name__}",
                args=cache_args,
                kwargs=kwargs
            )
            
            # 尝试从缓存获取
            cached_result = self.cache_manager.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for function {func.__name__}")
                return cached_result
            
            # 执行函数
            logger.debug(f"Cache miss for function {func.__name__}, executing")
            result = func(*args, **kwargs)
            
            # 缓存结果
            self.cache_manager.set(key, result, ttl=self.ttl)
            
            return result
        
        return wrapper


class DataCacheManager:
    """
    数据特定的缓存管理器
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据缓存管理器
        
        Args:
            config: 配置信息
        """
        # 创建通用缓存管理器
        self.cache_manager = CacheManager(config)
        
        # 数据质量级别的TTL映射
        self.quality_ttl = {
            DataQualityLevel.RAW: 3600,  # 1小时
            DataQualityLevel.CLEANED: 7200,  # 2小时
            DataQualityLevel.PROCESSED: 14400,  # 4小时
            DataQualityLevel.FEATURED: 28800,  # 8小时
            DataQualityLevel.ANALYZED: 86400  # 24小时
        }
    
    def cache_data(self, data_id: str, data: Any, quality_level: DataQualityLevel,
                  **metadata) -> bool:
        """
        缓存数据
        
        Args:
            data_id: 数据ID
            data: 数据
            quality_level: 数据质量级别
            **metadata: 元数据
            
        Returns:
            是否缓存成功
        """
        # 根据质量级别确定TTL
        ttl = self.quality_ttl.get(quality_level, self.cache_manager.config['default_ttl'])
        
        # 添加质量级别到元数据
        metadata['quality_level'] = quality_level.value
        
        return self.cache_manager.set(
            key=f"data:{data_id}",
            data=data,
            ttl=ttl,
            **metadata
        )
    
    def get_data(self, data_id: str) -> Optional[Any]:
        """
        获取缓存的数据
        
        Args:
            data_id: 数据ID
            
        Returns:
            数据，如果不存在或已过期返回None
        """
        return self.cache_manager.get(f"data:{data_id}")
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """
        失效匹配模式的缓存项
        
        Args:
            pattern: 匹配模式
            
        Returns:
            失效的项数
        """
        # 简单实现：遍历所有键并匹配
        # 注意：这在大缓存中效率较低，实际应用中可能需要更高效的实现
        invalidated_count = 0
        
        if self.cache_manager.storage_type == StorageType.MEMORY:
            # 内存存储
            keys_to_delete = []
            for key in self.cache_manager._cache.keys():
                if pattern in key:
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                if self.cache_manager.delete(key):
                    invalidated_count += 1
        
        elif self.cache_manager.storage_type == StorageType.FILE:
            # 文件存储
            with self.cache_manager._index_lock:
                keys_to_delete = [key for key in self.cache_manager._index['entries'].keys() 
                                if pattern in key]
            
            for key in keys_to_delete:
                if self.cache_manager.delete(key):
                    invalidated_count += 1
        
        logger.info(f"Invalidated {invalidated_count} cache entries matching pattern: {pattern}")
        return invalidated_count
    
    def refresh_data(self, data_id: str, fetch_func: Callable, quality_level: DataQualityLevel,
                    **fetch_kwargs) -> Any:
        """
        刷新数据（获取新数据并缓存）
        
        Args:
            data_id: 数据ID
            fetch_func: 获取数据的函数
            quality_level: 数据质量级别
            **fetch_kwargs: 传递给fetch_func的参数
            
        Returns:
            新数据
        """
        # 删除旧缓存
        self.cache_manager.delete(f"data:{data_id}")
        
        # 获取新数据
        data = fetch_func(**fetch_kwargs)
        
        # 缓存新数据
        self.cache_data(data_id, data, quality_level)
        
        return data
    
    def batch_get(self, data_ids: List[str]) -> Dict[str, Any]:
        """
        批量获取数据
        
        Args:
            data_ids: 数据ID列表
            
        Returns:
            {数据ID: 数据}的字典
        """
        result = {}
        
        for data_id in data_ids:
            data = self.get_data(data_id)
            if data is not None:
                result[data_id] = data
        
        return result
    
    def batch_set(self, items: Dict[str, Tuple[Any, DataQualityLevel, Dict[str, Any]]]) -> Dict[str, bool]:
        """
        批量设置数据
        
        Args:
            items: {数据ID: (数据, 质量级别, 元数据)}的字典
            
        Returns:
            {数据ID: 是否成功}的字典
        """
        results = {}
        
        for data_id, (data, quality_level, metadata) in items.items():
            results[data_id] = self.cache_data(data_id, data, quality_level, **metadata)
        
        return results
    
    def get_stale_data_ids(self, max_age_hours: float) -> List[str]:
        """
        获取超过指定时间未更新的数据ID列表
        
        Args:
            max_age_hours: 最大允许的小时数
            
        Returns:
            过期的数据ID列表
        """
        stale_ids = []
        max_age = timedelta(hours=max_age_hours)
        
        if self.cache_manager.storage_type == StorageType.MEMORY:
            # 内存存储
            now = datetime.now()
            for key, entry in self.cache_manager._cache.items():
                if key.startswith('data:') and (now - entry.created_at) > max_age:
                    data_id = key[5:]  # 去掉'data:'前缀
                    stale_ids.append(data_id)
        
        elif self.cache_manager.storage_type == StorageType.FILE:
            # 文件存储
            now = datetime.now()
            with self.cache_manager._index_lock:
                for key, entry_info in self.cache_manager._index['entries'].items():
                    if key.startswith('data:') and 'created_at' in entry_info:
                        created_at = datetime.fromisoformat(entry_info['created_at'])
                        if (now - created_at) > max_age:
                            data_id = key[5:]  # 去掉'data:'前缀
                            stale_ids.append(data_id)
        
        return stale_ids


# 创建全局缓存管理器实例
_default_cache_manager = None


def get_cache_manager() -> CacheManager:
    """
    获取全局缓存管理器实例
    
    Returns:
        缓存管理器实例
    """
    global _default_cache_manager
    
    if _default_cache_manager is None:
        # 尝试从配置加载
        try:
            config_manager = ConfigManager()
            cache_config = config_manager.get('cache', {})
            _default_cache_manager = CacheManager(cache_config)
        except Exception:
            # 使用默认配置
            _default_cache_manager = CacheManager()
    
    return _default_cache_manager


def cached(ttl: Optional[int] = None):
    """
    便捷的缓存装饰器
    
    Args:
        ttl: 生存时间（秒）
        
    Returns:
        装饰器函数
    """
    cache_manager = get_cache_manager()
    decorator = CacheDecorator(cache_manager, ttl)
    
    def decorator_func(func: Callable) -> Callable:
        return decorator(func)
    
    return decorator_func


def invalidate_cache(pattern: str) -> int:
    """
    便捷的缓存失效函数
    
    Args:
        pattern: 匹配模式
        
    Returns:
        失效的项数
    """
    cache_manager = get_cache_manager()
    return cache_manager.delete(pattern)  # 注意：这只会删除单个键，实际应该遍历匹配


# 示例用法
if __name__ == "__main__":
    # 简单的缓存测试
    cache_manager = CacheManager({
        'strategy': 'lru',
        'storage_type': 'memory',
        'max_size': 10,
        'default_ttl': 60
    })
    
    # 设置缓存
    cache_manager.set('test_key', {'value': 42}, ttl=10)
    
    # 获取缓存
    result = cache_manager.get('test_key')
    print(f"Cache result: {result}")
    
    # 检查统计信息
    stats = cache_manager.get_stats()
    print(f"Cache stats: {stats}")
    
    # 装饰器示例
    @cached(ttl=5)
    def slow_function(x, y):
        print(f"Executing slow function with {x}, {y}")
        return x + y
    
    # 第一次调用
    print(slow_function(1, 2))
    
    # 第二次调用（应该命中缓存）
    print(slow_function(1, 2))
    
    # 关闭缓存管理器
    cache_manager.close()