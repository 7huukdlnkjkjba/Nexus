#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志工具模块 (Logger)
提供分级日志系统，支持文件和控制台输出
"""

import logging
import os
import sys
import time
import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Dict, Any, Union


class LoggerManager:
    """
    日志管理器，提供分级日志功能
    """
    
    # 默认日志配置
    DEFAULT_CONFIG = {
        'console_level': 'INFO',
        'file_level': 'DEBUG',
        'log_dir': 'logs',
        'max_bytes': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5,
        'rotation_when': 'midnight',
        'rotation_interval': 1,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    # 日志级别映射
    LEVEL_MAP = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'WARN': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化日志管理器
        
        Args:
            config: 日志配置字典
        """
        # 合并配置
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # 创建日志目录
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # 日志格式器
        self.formatter = logging.Formatter(
            fmt=self.config['format'],
            datefmt=self.config['datefmt']
        )
        
        # 已初始化的日志记录器
        self.loggers: Dict[str, logging.Logger] = {}
        
        # 根日志记录器配置
        self._configure_root_logger()
    
    def _configure_root_logger(self) -> None:
        """
        配置根日志记录器
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # 捕获所有级别的日志
        
        # 移除现有的处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 添加控制台处理器
        console_handler = self._create_console_handler()
        root_logger.addHandler(console_handler)
        
        # 添加文件处理器
        file_handler = self._create_file_handler('nexus')
        root_logger.addHandler(file_handler)
    
    def _create_console_handler(self) -> logging.StreamHandler:
        """
        创建控制台日志处理器
        
        Returns:
            控制台日志处理器
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.LEVEL_MAP[self.config['console_level']])
        console_handler.setFormatter(self.formatter)
        return console_handler
    
    def _create_file_handler(self, logger_name: str) -> Union[RotatingFileHandler, TimedRotatingFileHandler]:
        """
        创建文件日志处理器
        
        Args:
            logger_name: 日志记录器名称
            
        Returns:
            文件日志处理器
        """
        log_file = os.path.join(self.config['log_dir'], f'{logger_name}.log')
        
        # 可以根据需要选择不同的轮转策略
        if self.config.get('use_size_rotation', False):
            # 基于大小的轮转
            handler = RotatingFileHandler(
                log_file,
                maxBytes=self.config['max_bytes'],
                backupCount=self.config['backup_count'],
                encoding='utf-8'
            )
        else:
            # 基于时间的轮转
            handler = TimedRotatingFileHandler(
                log_file,
                when=self.config['rotation_when'],
                interval=self.config['rotation_interval'],
                backupCount=self.config['backup_count'],
                encoding='utf-8'
            )
        
        handler.setLevel(self.LEVEL_MAP[self.config['file_level']])
        handler.setFormatter(self.formatter)
        return handler
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        获取指定名称的日志记录器
        
        Args:
            name: 日志记录器名称
            
        Returns:
            日志记录器实例
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            
            # 移除现有的处理器（避免重复）
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # 添加控制台处理器
            console_handler = self._create_console_handler()
            logger.addHandler(console_handler)
            
            # 添加文件处理器
            file_handler = self._create_file_handler(name)
            logger.addHandler(file_handler)
            
            # 避免向上传播到根日志记录器
            logger.propagate = False
            
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def set_level(self, logger_name: Optional[str], level: str) -> None:
        """
        设置日志级别
        
        Args:
            logger_name: 日志记录器名称，None表示设置控制台和文件默认级别
            level: 日志级别字符串
        """
        if level not in self.LEVEL_MAP:
            raise ValueError(f"Invalid log level: {level}")
        
        logging_level = self.LEVEL_MAP[level]
        
        if logger_name is None:
            # 更新默认级别
            self.config['console_level'] = level
            self.config['file_level'] = level
            
            # 更新所有处理器
            for logger in self.loggers.values():
                for handler in logger.handlers:
                    if isinstance(handler, logging.StreamHandler):
                        handler.setLevel(logging_level)
                    elif isinstance(handler, (RotatingFileHandler, TimedRotatingFileHandler)):
                        handler.setLevel(logging_level)
        else:
            # 更新指定日志记录器的级别
            if logger_name in self.loggers:
                logger = self.loggers[logger_name]
                logger.setLevel(logging_level)
    
    def add_handler(self, logger_name: str, handler: logging.Handler) -> None:
        """
        为指定日志记录器添加额外的处理器
        
        Args:
            logger_name: 日志记录器名称
            handler: 日志处理器
        """
        logger = self.get_logger(logger_name)
        handler.setFormatter(self.formatter)
        logger.addHandler(handler)
    
    def log_with_context(self, logger_name: str, level: str, message: str, 
                        context: Optional[Dict[str, Any]] = None) -> None:
        """
        记录带有上下文信息的日志
        
        Args:
            logger_name: 日志记录器名称
            level: 日志级别
            message: 日志消息
            context: 上下文信息字典
        """
        logger = self.get_logger(logger_name)
        
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            full_message = f"{message} [{context_str}]"
        else:
            full_message = message
        
        # 根据级别记录日志
        if level == 'DEBUG':
            logger.debug(full_message)
        elif level == 'INFO':
            logger.info(full_message)
        elif level in ('WARNING', 'WARN'):
            logger.warning(full_message)
        elif level == 'ERROR':
            logger.error(full_message)
        elif level == 'CRITICAL':
            logger.critical(full_message)
    
    def create_performance_logger(self) -> logging.Logger:
        """
        创建性能日志记录器
        
        Returns:
            性能日志记录器
        """
        perf_logger = self.get_logger('nexus_performance')
        
        # 自定义性能日志格式
        perf_formatter = logging.Formatter(
            '%(asctime)s - PERFORMANCE - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S.%f'
        )
        
        # 更新处理器格式
        for handler in perf_logger.handlers:
            handler.setFormatter(perf_formatter)
        
        return perf_logger


# 创建全局日志管理器实例
_logger_manager = None


def init_logger(config: Optional[Dict[str, Any]] = None) -> LoggerManager:
    """
    初始化全局日志管理器
    
    Args:
        config: 日志配置
        
    Returns:
        日志管理器实例
    """
    global _logger_manager
    _logger_manager = LoggerManager(config)
    return _logger_manager


def get_logger(name: str = 'nexus') -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器实例
    """
    global _logger_manager
    if _logger_manager is None:
        init_logger()
    return _logger_manager.get_logger(name)


def log_performance(logger: logging.Logger, operation: str, start_time: float) -> None:
    """
    记录性能指标
    
    Args:
        logger: 日志记录器
        operation: 操作名称
        start_time: 开始时间戳
    """
    duration = time.time() - start_time
    logger.info(f"Operation: {operation} | Duration: {duration:.4f}s")


class LoggingContext:
    """
    日志上下文管理器，用于临时更改日志级别
    """
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.old_level = logger.level
    
    def __enter__(self):
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


# 初始化默认日志记录器
default_logger = get_logger()

# 提供便捷的日志函数
def debug(msg, *args, **kwargs):
    default_logger.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    default_logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    default_logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    default_logger.error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    default_logger.critical(msg, *args, **kwargs)

def exception(msg, *args, **kwargs):
    default_logger.exception(msg, *args, **kwargs)