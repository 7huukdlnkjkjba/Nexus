#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
性能指标模块 (Metrics)
提供性能指标的收集、聚合和可视化功能
"""

import time
import json
import os
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from .logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """
    指标收集器，用于收集和存储各种性能指标
    """
    
    def __init__(self, name: str = 'default', 
                 buffer_size: int = 10000,
                 enable_file_export: bool = True,
                 export_dir: str = 'metrics'):
        """
        初始化指标收集器
        
        Args:
            name: 收集器名称
            buffer_size: 内存缓冲区大小
            enable_file_export: 是否启用文件导出
            export_dir: 导出目录
        """
        self.name = name
        self.buffer_size = buffer_size
        self.enable_file_export = enable_file_export
        self.export_dir = export_dir
        
        # 指标数据
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}
        
        # 聚合数据
        self._aggregates: Dict[str, Dict[str, Any]] = {}
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 确保导出目录存在
        if self.enable_file_export:
            os.makedirs(self.export_dir, exist_ok=True)
        
        logger.info(f"MetricsCollector initialized: {name}")
    
    def collect(self, metric_name: str, value: Union[float, int], 
                tags: Optional[Dict[str, Any]] = None, 
                timestamp: Optional[float] = None) -> None:
        """
        收集一个指标数据点
        
        Args:
            metric_name: 指标名称
            value: 指标值
            tags: 标签字典
            timestamp: 时间戳，默认为当前时间
        """
        with self._lock:
            if metric_name not in self._metrics:
                self._metrics[metric_name] = []
            
            # 创建数据点
            data_point = {
                'timestamp': timestamp or time.time(),
                'value': value,
                'tags': tags or {}
            }
            
            # 添加到缓冲区
            self._metrics[metric_name].append(data_point)
            
            # 如果缓冲区过大，移除旧数据
            if len(self._metrics[metric_name]) > self.buffer_size:
                self._metrics[metric_name] = self._metrics[metric_name][-self.buffer_size:]
            
            # 更新聚合数据
            self._update_aggregates(metric_name, value)
    
    def _update_aggregates(self, metric_name: str, value: Union[float, int]) -> None:
        """
        更新聚合数据
        
        Args:
            metric_name: 指标名称
            value: 指标值
        """
        if metric_name not in self._aggregates:
            self._aggregates[metric_name] = {
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'values': []
            }
        
        agg = self._aggregates[metric_name]
        agg['count'] += 1
        agg['sum'] += value
        agg['min'] = min(agg['min'], value)
        agg['max'] = max(agg['max'], value)
        agg['values'].append(value)
        
        # 保持聚合样本数量合理
        if len(agg['values']) > 1000:  # 只保留最近的1000个值用于计算标准差
            agg['values'] = agg['values'][-1000:]
    
    def collect_timer(self, metric_name: str, start_time: float, 
                     tags: Optional[Dict[str, Any]] = None) -> float:
        """
        收集时间指标
        
        Args:
            metric_name: 指标名称
            start_time: 开始时间戳
            tags: 标签字典
            
        Returns:
            耗时（秒）
        """
        duration = time.time() - start_time
        self.collect(metric_name, duration, tags)
        return duration
    
    def get_metrics(self, metric_name: str) -> List[Dict[str, Any]]:
        """
        获取指定指标的所有数据点
        
        Args:
            metric_name: 指标名称
            
        Returns:
            数据点列表
        """
        with self._lock:
            return self._metrics.get(metric_name, []).copy()
    
    def get_aggregate(self, metric_name: str) -> Dict[str, Any]:
        """
        获取指定指标的聚合数据
        
        Args:
            metric_name: 指标名称
            
        Returns:
            聚合数据字典
        """
        with self._lock:
            if metric_name not in self._aggregates:
                return None
            
            agg = self._aggregates[metric_name].copy()
            
            # 计算平均值和标准差
            if agg['count'] > 0:
                agg['avg'] = agg['sum'] / agg['count']
                if len(agg['values']) > 1:
                    agg['std'] = np.std(agg['values'])
                else:
                    agg['std'] = 0.0
            
            return agg
    
    def get_all_metrics_names(self) -> List[str]:
        """
        获取所有收集的指标名称
        
        Returns:
            指标名称列表
        """
        with self._lock:
            return list(self._metrics.keys())
    
    def export_to_file(self, metric_name: str, format: str = 'json') -> str:
        """
        导出指标数据到文件
        
        Args:
            metric_name: 指标名称
            format: 导出格式 ('json' 或 'csv')
            
        Returns:
            导出的文件路径
        """
        if not self.enable_file_export:
            raise ValueError("File export is disabled")
        
        metrics = self.get_metrics(metric_name)
        if not metrics:
            raise ValueError(f"No metrics found for {metric_name}")
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{metric_name}_{timestamp}.{format}"
        filepath = os.path.join(self.export_dir, filename)
        
        try:
            if format == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
            elif format == 'csv':
                # 转换为DataFrame并导出
                df = pd.DataFrame([{
                    'timestamp': m['timestamp'],
                    'value': m['value'],
                    **m['tags']
                } for m in metrics])
                df.to_csv(filepath, index=False, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported metrics to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise
    
    def clear(self, metric_name: Optional[str] = None) -> None:
        """
        清除指标数据
        
        Args:
            metric_name: 指标名称，如果为None则清除所有指标
        """
        with self._lock:
            if metric_name:
                if metric_name in self._metrics:
                    del self._metrics[metric_name]
                if metric_name in self._aggregates:
                    del self._aggregates[metric_name]
            else:
                self._metrics.clear()
                self._aggregates.clear()
    
    def plot_metric(self, metric_name: str, output_file: Optional[str] = None,
                   show: bool = False, **kwargs) -> plt.Figure:
        """
        绘制指标图表
        
        Args:
            metric_name: 指标名称
            output_file: 输出文件路径
            show: 是否显示图表
            **kwargs: 传递给plot的额外参数
            
        Returns:
            Matplotlib Figure对象
        """
        metrics = self.get_metrics(metric_name)
        if not metrics:
            raise ValueError(f"No metrics found for {metric_name}")
        
        # 转换为DataFrame
        timestamps = [m['timestamp'] for m in metrics]
        values = [m['value'] for m in metrics]
        
        # 创建图表
        plt.figure(figsize=kwargs.get('figsize', (12, 6)))
        
        # 绘制折线图
        plt.plot(timestamps, values, marker='o' if len(values) < 100 else None,
                linestyle='-', **kwargs)
        
        # 设置标题和标签
        plt.title(f'Metric: {metric_name}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        
        # 格式化时间轴
        try:
            plt.gcf().autofmt_xdate()
        except:
            pass
        
        # 保存或显示
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf()
    
    def plot_distribution(self, metric_name: str, output_file: Optional[str] = None,
                         show: bool = False, bins: int = 30, **kwargs) -> plt.Figure:
        """
        绘制指标分布直方图
        
        Args:
            metric_name: 指标名称
            output_file: 输出文件路径
            show: 是否显示图表
            bins: 直方图分箱数
            **kwargs: 传递给hist的额外参数
            
        Returns:
            Matplotlib Figure对象
        """
        metrics = self.get_metrics(metric_name)
        if not metrics:
            raise ValueError(f"No metrics found for {metric_name}")
        
        # 提取值
        values = [m['value'] for m in metrics]
        
        # 创建图表
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        
        # 绘制直方图
        plt.hist(values, bins=bins, alpha=0.7, **kwargs)
        
        # 添加统计信息
        agg = self.get_aggregate(metric_name)
        if agg:
            plt.axvline(agg['avg'], color='r', linestyle='--', label=f'Mean: {agg["avg"]:.2f}')
            plt.axvline(agg['min'], color='g', linestyle='-.', label=f'Min: {agg["min"]:.2f}')
            plt.axvline(agg['max'], color='b', linestyle='-.', label=f'Max: {agg["max"]:.2f}')
            plt.legend()
        
        plt.title(f'Distribution of {metric_name}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 保存或显示
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plot saved to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf()
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        生成指标报告
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            报告数据字典
        """
        report = {
            'timestamp': time.time(),
            'collector_name': self.name,
            'metrics': {}
        }
        
        # 收集每个指标的聚合数据
        for metric_name in self.get_all_metrics_names():
            agg = self.get_aggregate(metric_name)
            if agg:
                report['metrics'][metric_name] = agg
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Report saved to {output_file}")
        
        return report


class MetricsManager:
    """
    指标管理器，管理多个指标收集器
    """
    
    def __init__(self):
        """
        初始化指标管理器
        """
        self._collectors: Dict[str, MetricsCollector] = {}
        self._lock = threading.RLock()
    
    def get_collector(self, name: str, **kwargs) -> MetricsCollector:
        """
        获取或创建指标收集器
        
        Args:
            name: 收集器名称
            **kwargs: 创建收集器的参数
            
        Returns:
            指标收集器实例
        """
        with self._lock:
            if name not in self._collectors:
                self._collectors[name] = MetricsCollector(name, **kwargs)
            return self._collectors[name]
    
    def collect(self, collector_name: str, metric_name: str, value: Union[float, int],
               tags: Optional[Dict[str, Any]] = None) -> None:
        """
        收集指标
        
        Args:
            collector_name: 收集器名称
            metric_name: 指标名称
            value: 指标值
            tags: 标签
        """
        collector = self.get_collector(collector_name)
        collector.collect(metric_name, value, tags)
    
    def collect_timer(self, collector_name: str, metric_name: str, start_time: float,
                     tags: Optional[Dict[str, Any]] = None) -> float:
        """
        收集时间指标
        
        Args:
            collector_name: 收集器名称
            metric_name: 指标名称
            start_time: 开始时间
            tags: 标签
            
        Returns:
            耗时
        """
        collector = self.get_collector(collector_name)
        return collector.collect_timer(metric_name, start_time, tags)
    
    def clear_collector(self, name: str) -> None:
        """
        清除指定的收集器
        
        Args:
            name: 收集器名称
        """
        with self._lock:
            if name in self._collectors:
                del self._collectors[name]
    
    def clear_all(self) -> None:
        """
        清除所有收集器
        """
        with self._lock:
            self._collectors.clear()
    
    def get_all_collectors(self) -> List[str]:
        """
        获取所有收集器名称
        
        Returns:
            收集器名称列表
        """
        with self._lock:
            return list(self._collectors.keys())


# 装饰器函数
def measure_performance(collector_name: str = 'default', 
                       metric_name: Optional[str] = None):
    """
    性能测量装饰器
    
    Args:
        collector_name: 收集器名称
        metric_name: 指标名称，默认为函数名
        
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        actual_metric_name = metric_name or func.__name__
        
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # 收集性能指标
                get_metrics_manager().collect_timer(
                    collector_name, actual_metric_name,
                    start_time, tags={'function': func.__name__}
                )
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


def track_memory_usage(collector_name: str = 'default',
                      metric_name: str = 'memory_usage'):
    """
    内存使用跟踪装饰器
    
    Args:
        collector_name: 收集器名称
        metric_name: 指标名称
        
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # 尝试导入内存监控模块
            try:
                import psutil
                import os
                
                # 获取当前进程
                process = psutil.Process(os.getpid())
                
                # 执行前内存
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # 执行函数
                result = func(*args, **kwargs)
                
                # 执行后内存
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                
                # 收集指标
                get_metrics_manager().collect(
                    collector_name, mem_after,
                    tags={
                        'function': func.__name__,
                        'mem_before': mem_before,
                        'mem_diff': mem_after - mem_before
                    }
                )
                
                return result
                
            except ImportError:
                logger.warning("psutil not installed, memory tracking disabled")
                return func(*args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


# 全局指标管理器实例
_metrics_manager = None

def init_metrics_manager() -> MetricsManager:
    """
    初始化全局指标管理器
    
    Returns:
        指标管理器实例
    """
    global _metrics_manager
    _metrics_manager = MetricsManager()
    return _metrics_manager

def get_metrics_manager() -> MetricsManager:
    """
    获取全局指标管理器实例
    
    Returns:
        指标管理器实例
    """
    global _metrics_manager
    if _metrics_manager is None:
        init_metrics_manager()
    return _metrics_manager

# 便捷函数
def collect_metric(collector_name: str, metric_name: str, value: Union[float, int],
                  tags: Optional[Dict[str, Any]] = None) -> None:
    """
    便捷收集指标
    
    Args:
        collector_name: 收集器名称
        metric_name: 指标名称
        value: 指标值
        tags: 标签
    """
    get_metrics_manager().collect(collector_name, metric_name, value, tags)

def start_timer(collector_name: str = 'default',
               metric_name: str = 'operation_time') -> Dict[str, Any]:
    """
    开始计时
    
    Args:
        collector_name: 收集器名称
        metric_name: 指标名称
        
    Returns:
        计时上下文
    """
    return {
        'collector_name': collector_name,
        'metric_name': metric_name,
        'start_time': time.time()
    }

def end_timer(timer_context: Dict[str, Any], tags: Optional[Dict[str, Any]] = None) -> float:
    """
    结束计时
    
    Args:
        timer_context: 计时上下文
        tags: 标签
        
    Returns:
        耗时
    """
    return get_metrics_manager().collect_timer(
        timer_context['collector_name'],
        timer_context['metric_name'],
        timer_context['start_time'],
        tags
    )


# 预定义的常用指标收集器
def get_performance_collector() -> MetricsCollector:
    """
    获取性能指标收集器
    
    Returns:
        性能指标收集器
    """
    return get_metrics_manager().get_collector('performance')

def get_simulation_collector() -> MetricsCollector:
    """
    获取模拟指标收集器
    
    Returns:
        模拟指标收集器
    """
    return get_metrics_manager().get_collector('simulation')

def get_system_collector() -> MetricsCollector:
    """
    获取系统指标收集器
    
    Returns:
        系统指标收集器
    """
    return get_metrics_manager().get_collector('system')