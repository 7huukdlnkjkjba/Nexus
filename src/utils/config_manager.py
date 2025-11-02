#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块 (Config Manager)
提供YAML配置文件的加载、解析和热更新功能
"""

import os
import yaml
import json
import copy
import threading
import time
from typing import Dict, Any, Optional, List
import logging

from .logger import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """
    配置管理器，负责加载和管理配置文件
    """
    
    def __init__(self, config_dir: str = 'config', auto_reload: bool = True,
                 reload_interval: int = 30):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
            auto_reload: 是否自动重新加载配置
            reload_interval: 自动重新加载间隔（秒）
        """
        self.config_dir = config_dir
        self.auto_reload = auto_reload
        self.reload_interval = reload_interval
        
        # 配置缓存
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        self._file_mtimes: Dict[str, float] = {}
        self._default_values: Dict[str, Any] = {}
        
        # 线程锁，保证线程安全
        self._lock = threading.RLock()
        
        # 自动重载线程
        self._reload_thread = None
        self._stop_event = threading.Event()
        
        # 确保配置目录存在
        os.makedirs(self.config_dir, exist_ok=True)
        
        # 启动自动重载线程
        if self.auto_reload:
            self._start_auto_reload()
    
    def _start_auto_reload(self) -> None:
        """
        启动自动重载线程
        """
        def reload_loop():
            while not self._stop_event.is_set():
                try:
                    self._stop_event.wait(self.reload_interval)
                    if not self._stop_event.is_set():
                        self.reload_all()
                except Exception as e:
                    logger.error(f"Error in auto reload: {e}")
        
        self._reload_thread = threading.Thread(target=reload_loop, daemon=True)
        self._reload_thread.start()
    
    def stop(self) -> None:
        """
        停止配置管理器和自动重载线程
        """
        self._stop_event.set()
        if self._reload_thread:
            self._reload_thread.join(timeout=5)
    
    def load_config(self, config_name: str, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        加载指定的配置文件
        
        Args:
            config_name: 配置文件名（不含扩展名）
            default_config: 默认配置
            
        Returns:
            配置字典
        """
        with self._lock:
            # 尝试多种配置文件格式
            extensions = ['.yaml', '.yml', '.json']
            config_path = None
            
            for ext in extensions:
                path = os.path.join(self.config_dir, f'{config_name}{ext}')
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path is None:
                logger.warning(f"Config file not found for {config_name}")
                return default_config or {}
            
            # 检查文件是否被修改
            file_mtime = os.path.getmtime(config_path)
            if config_name in self._file_mtimes and self._file_mtimes[config_name] == file_mtime:
                # 使用缓存的配置
                return copy.deepcopy(self._config_cache.get(config_name, default_config or {}))
            
            # 更新修改时间
            self._file_mtimes[config_name] = file_mtime
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.endswith('.json'):
                        config = json.load(f)
                    else:
                        config = yaml.safe_load(f)
                
                # 合并默认配置
                if default_config:
                    config = self._merge_configs(config or {}, default_config)
                
                # 缓存配置
                self._config_cache[config_name] = config
                logger.info(f"Loaded config from {config_path}")
                
                return copy.deepcopy(config)
                
            except Exception as e:
                logger.error(f"Error loading config {config_path}: {e}")
                # 返回缓存的配置或默认配置
                return copy.deepcopy(self._config_cache.get(config_name, default_config or {}))
    
    def _merge_configs(self, config: Dict[str, Any], default: Dict[str, Any]) -> Dict[str, Any]:
        """
        递归合并配置字典
        
        Args:
            config: 用户配置
            default: 默认配置
            
        Returns:
            合并后的配置
        """
        merged = default.copy()
        
        for key, value in config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                merged[key] = self._merge_configs(value, merged[key])
            else:
                # 直接覆盖
                merged[key] = value
        
        return merged
    
    def get_config(self, config_name: str, key_path: Optional[str] = None,
                  default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            config_name: 配置文件名
            key_path: 配置键路径，使用点号分隔，如 "database.host"
            default: 默认值
            
        Returns:
            配置值
        """
        # 加载配置
        config = self.load_config(config_name)
        
        # 如果没有指定键路径，返回整个配置
        if key_path is None:
            return config
        
        # 解析键路径
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value
        except Exception as e:
            logger.error(f"Error getting config value {key_path}: {e}")
            return default
    
    def set_config(self, config_name: str, key_path: str, value: Any) -> bool:
        """
        设置配置值（仅内存中，不会写入文件）
        
        Args:
            config_name: 配置文件名
            key_path: 配置键路径
            value: 配置值
            
        Returns:
            是否设置成功
        """
        with self._lock:
            # 确保配置已加载
            if config_name not in self._config_cache:
                self.load_config(config_name)
            
            if config_name not in self._config_cache:
                self._config_cache[config_name] = {}
            
            config = self._config_cache[config_name]
            keys = key_path.split('.')
            
            # 导航到目标位置
            current = config
            for key in keys[:-1]:
                if key not in current or not isinstance(current[key], dict):
                    current[key] = {}
                current = current[key]
            
            # 设置值
            current[keys[-1]] = value
            logger.info(f"Set config {config_name}:{key_path} = {value}")
            return True
    
    def reload_all(self) -> None:
        """
        重新加载所有配置文件
        """
        with self._lock:
            config_names = list(self._config_cache.keys())
            logger.info(f"Reloading {len(config_names)} config files")
            
            for config_name in config_names:
                self.load_config(config_name)
    
    def reload(self, config_name: str) -> Dict[str, Any]:
        """
        重新加载指定的配置文件
        
        Args:
            config_name: 配置文件名
            
        Returns:
            重新加载后的配置
        """
        with self._lock:
            if config_name in self._file_mtimes:
                del self._file_mtimes[config_name]
            if config_name in self._config_cache:
                del self._config_cache[config_name]
            
            return self.load_config(config_name)
    
    def save_config(self, config_name: str, config: Dict[str, Any],
                   format: str = 'yaml') -> bool:
        """
        保存配置到文件
        
        Args:
            config_name: 配置文件名
            config: 配置字典
            format: 文件格式 ('yaml' 或 'json')
            
        Returns:
            是否保存成功
        """
        ext = '.yaml' if format == 'yaml' else '.json'
        config_path = os.path.join(self.config_dir, f'{config_name}{ext}')
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if format == 'yaml':
                    yaml.dump(config, f, default_flow_style=False,
                             allow_unicode=True, sort_keys=False)
                else:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            
            # 更新缓存
            with self._lock:
                self._config_cache[config_name] = config
                self._file_mtimes[config_name] = os.path.getmtime(config_path)
            
            logger.info(f"Saved config to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            return False
    
    def validate_config(self, config_name: str, schema: Dict[str, Any]) -> bool:
        """
        验证配置是否符合 schema
        
        Args:
            config_name: 配置文件名
            schema: 配置 schema，定义必需字段和类型
            
        Returns:
            配置是否有效
        """
        config = self.load_config(config_name)
        return self._validate_config_against_schema(config, schema)
    
    def _validate_config_against_schema(self, config: Dict[str, Any], 
                                      schema: Dict[str, Any]) -> bool:
        """
        递归验证配置是否符合 schema
        
        Args:
            config: 配置字典
            schema: schema 字典
            
        Returns:
            配置是否有效
        """
        # 检查必需字段
        required = schema.get('required', [])
        for field in required:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False
        
        # 检查字段类型
        properties = schema.get('properties', {})
        for field, field_schema in properties.items():
            if field in config:
                expected_type = field_schema.get('type')
                if expected_type:
                    # 处理类型检查
                    if expected_type == 'string' and not isinstance(config[field], str):
                        logger.error(f"Field {field} should be string, got {type(config[field])}")
                        return False
                    elif expected_type == 'number' and not isinstance(config[field], (int, float)):
                        logger.error(f"Field {field} should be number, got {type(config[field])}")
                        return False
                    elif expected_type == 'integer' and not isinstance(config[field], int):
                        logger.error(f"Field {field} should be integer, got {type(config[field])}")
                        return False
                    elif expected_type == 'boolean' and not isinstance(config[field], bool):
                        logger.error(f"Field {field} should be boolean, got {type(config[field])}")
                        return False
                    elif expected_type == 'array' and not isinstance(config[field], list):
                        logger.error(f"Field {field} should be array, got {type(config[field])}")
                        return False
                    elif expected_type == 'object' and not isinstance(config[field], dict):
                        logger.error(f"Field {field} should be object, got {type(config[field])}")
                        return False
                
                # 递归验证嵌套对象
                if expected_type == 'object' and 'properties' in field_schema:
                    if not self._validate_config_against_schema(
                            config[field], field_schema):
                        return False
        
        return True
    
    def get_environment_config(self, prefix: str = 'NEXUS_') -> Dict[str, Any]:
        """
        从环境变量加载配置
        
        Args:
            prefix: 环境变量前缀
            
        Returns:
            环境变量配置字典
        """
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # 移除前缀并转换为小写
                config_key = key[len(prefix):].lower()
                # 转换类型
                config[config_key] = self._convert_env_value(value)
        
        return config
    
    def _convert_env_value(self, value: str) -> Any:
        """
        转换环境变量值为适当的类型
        
        Args:
            value: 环境变量字符串值
            
        Returns:
            转换后的值
        """
        # 尝试转换为布尔值
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        
        # 尝试转换为整数
        try:
            return int(value)
        except ValueError:
            pass
        
        # 尝试转换为浮点数
        try:
            return float(value)
        except ValueError:
            pass
        
        # 尝试转换为列表
        if value.startswith('[') and value.endswith(']'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # 默认返回字符串
        return value
    
    def get_all_config_names(self) -> List[str]:
        """
        获取所有可用的配置文件名
        
        Returns:
            配置文件名列表
        """
        config_names = set()
        
        if not os.path.exists(self.config_dir):
            return []
        
        for filename in os.listdir(self.config_dir):
            if filename.endswith(('.yaml', '.yml', '.json')):
                config_name = os.path.splitext(filename)[0]
                config_names.add(config_name)
        
        return list(config_names)


# 创建全局配置管理器实例
_config_manager = None

def init_config_manager(config_dir: str = 'config', **kwargs) -> ConfigManager:
    """
    初始化全局配置管理器
    
    Args:
        config_dir: 配置文件目录
        **kwargs: 其他参数
        
    Returns:
        配置管理器实例
    """
    global _config_manager
    _config_manager = ConfigManager(config_dir, **kwargs)
    return _config_manager

def get_config_manager() -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Returns:
        配置管理器实例
    """
    global _config_manager
    if _config_manager is None:
        init_config_manager()
    return _config_manager

# 便捷函数
def get_config(config_name: str, key_path: Optional[str] = None,
              default: Any = None) -> Any:
    """
    便捷获取配置值
    
    Args:
        config_name: 配置文件名
        key_path: 配置键路径
        default: 默认值
        
    Returns:
        配置值
    """
    return get_config_manager().get_config(config_name, key_path, default)

def load_config(config_name: str, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    便捷加载配置文件
    
    Args:
        config_name: 配置文件名
        default_config: 默认配置
        
    Returns:
        配置字典
    """
    return get_config_manager().load_config(config_name, default_config)

def reload_config(config_name: str) -> Dict[str, Any]:
    """
    便捷重新加载配置文件
    
    Args:
        config_name: 配置文件名
        
    Returns:
        重新加载后的配置
    """
    return get_config_manager().reload(config_name)


def stop_config_manager():
    """
    停止配置管理器
    """
    global _config_manager
    if _config_manager:
        _config_manager.stop()
        _config_manager = None


# 默认配置模板
DEFAULT_CONFIG_SCHEMA = {
    'required': [],
    'properties': {
        'debug': {
            'type': 'boolean',
            'default': False
        },
        'log_level': {
            'type': 'string',
            'enum': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            'default': 'INFO'
        }
    }
}