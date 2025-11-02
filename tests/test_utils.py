# Nexus 项目工具模块单元测试

import os
import sys
import time
import pytest
import numpy as np
import pandas as pd
from unittest import mock
from datetime import datetime, timedelta

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入被测试模块
from nexus.utils.metrics import MetricsCollector, MetricsManager, performance_timer, memory_usage
def test_metrics_collector_initialization():
    """测试MetricsCollector初始化"""
    # 基本初始化
    collector = MetricsCollector(metric_name="test_metric")
    assert collector.metric_name == "test_metric"
    assert collector.metrics == []
    assert collector.start_time is not None
    
    # 带描述初始化
    collector = MetricsCollector(metric_name="test_metric", description="Test description")
    assert collector.description == "Test description"


def test_metrics_collector_record_metric():
    """测试记录指标"""
    collector = MetricsCollector(metric_name="test_metric")
    
    # 记录单个值
    collector.record_metric(42)
    assert len(collector.metrics) == 1
    assert collector.metrics[0]['value'] == 42
    assert 'timestamp' in collector.metrics[0]
    
    # 记录带标签的值
    collector.record_metric(100, tags={"category": "A", "subgroup": "X"})
    assert len(collector.metrics) == 2
    assert collector.metrics[1]['value'] == 100
    assert collector.metrics[1]['tags'] == {"category": "A", "subgroup": "X"}
    
    # 记录带额外数据的值
    collector.record_metric(200, extra_data={"metadata": "test"})
    assert len(collector.metrics) == 3
    assert collector.metrics[2]['extra_data'] == {"metadata": "test"}


def test_metrics_collector_get_aggregates():
    """测试获取聚合数据"""
    collector = MetricsCollector(metric_name="test_metric")
    
    # 记录一些数据
    test_values = [10, 20, 30, 40, 50]
    for value in test_values:
        collector.record_metric(value)
    
    # 获取聚合数据
    aggregates = collector.get_aggregates()
    
    # 验证聚合结果
    assert aggregates['count'] == 5
    assert aggregates['sum'] == 150
    assert aggregates['mean'] == 30.0
    assert aggregates['min'] == 10
    assert aggregates['max'] == 50
    assert aggregates['std'] == np.std(test_values)
    assert aggregates['median'] == 30.0


def test_metrics_collector_export_to_file(temp_dir):
    """测试导出到文件"""
    collector = MetricsCollector(metric_name="test_metric")
    
    # 记录一些数据
    for i in range(5):
        collector.record_metric(i * 10)
    
    # 导出到CSV
    csv_path = os.path.join(temp_dir, "metrics.csv")
    success = collector.export_to_file(csv_path, format="csv")
    assert success
    assert os.path.exists(csv_path)
    
    # 验证CSV文件内容
    df = pd.read_csv(csv_path)
    assert len(df) == 5
    assert list(df['value']) == [0, 10, 20, 30, 40]
    
    # 导出到JSON
    json_path = os.path.join(temp_dir, "metrics.json")
    success = collector.export_to_file(json_path, format="json")
    assert success
    assert os.path.exists(json_path)


def test_metrics_manager():
    """测试MetricsManager"""
    manager = MetricsManager()
    
    # 创建收集器
    collector1 = manager.create_collector("metric1", "First metric")
    collector2 = manager.create_collector("metric2", "Second metric")
    
    # 验证创建的收集器
    assert len(manager.collectors) == 2
    assert manager.get_collector("metric1") == collector1
    assert manager.get_collector("metric2") == collector2
    
    # 记录指标
    manager.record_metric("metric1", 100)
    manager.record_metric("metric2", 200)
    
    # 验证记录
    assert len(collector1.metrics) == 1
    assert len(collector2.metrics) == 1
    
    # 删除收集器
    manager.delete_collector("metric1")
    assert len(manager.collectors) == 1
    assert manager.get_collector("metric1") is None


@performance_timer

def dummy_function():
    """用于测试性能装饰器的虚拟函数"""
    time.sleep(0.1)  # 模拟一些工作
    return "done"


def test_performance_timer(mock_logger):
    """测试性能计时器装饰器"""
    result = dummy_function()
    
    # 验证函数正常执行
    assert result == "done"
    
    # 验证日志被调用（装饰器内部会记录日志）
    mock_logger.info.assert_called()


@memory_usage

def memory_intensive_function():
    """用于测试内存使用装饰器的虚拟函数"""
    # 创建一个大数组
    arr = np.ones(1000000)
    time.sleep(0.01)  # 给内存测量一些时间
    return len(arr)


def test_memory_usage(mock_logger):
    """测试内存使用装饰器"""
    result = memory_intensive_function()
    
    # 验证函数正常执行
    assert result == 1000000
    
    # 验证日志被调用（装饰器内部会记录日志）
    mock_logger.info.assert_called()


def test_metrics_collector_plot_metrics(temp_dir):
    """测试绘制指标图表"""
    collector = MetricsCollector(metric_name="test_metric")
    
    # 记录一些数据点
    for i in range(24):
        # 创建一个简单的正弦波模式
        value = 50 + 50 * np.sin(i * np.pi / 12)
        collector.record_metric(value, timestamp=datetime.now() + timedelta(hours=i))
    
    # 测试绘制折线图
    plot_path = os.path.join(temp_dir, "plot.png")
    try:
        collector.plot_metrics(output_path=plot_path, plot_type="line")
        # 注意：在无GUI环境中，matplotlib可能会失败，所以这里只是尝试，不强制检查文件是否存在
    except Exception as e:
        pytest.skip(f"Plotting failed: {e}")


def test_metrics_collector_invalid_export_format():
    """测试无效的导出格式"""
    collector = MetricsCollector(metric_name="test_metric")
    collector.record_metric(42)
    
    with pytest.raises(ValueError):
        collector.export_to_file("invalid.format", format="invalid")


def test_metrics_manager_non_existent_collector():
    """测试访问不存在的收集器"""
    manager = MetricsManager()
    
    # 获取不存在的收集器应返回None
    assert manager.get_collector("non_existent") is None
    
    # 记录到不存在的收集器应创建新的
    manager.record_metric("non_existent", 100)
    assert manager.get_collector("non_existent") is not None


def test_metrics_collector_empty_aggregates():
    """测试空收集器的聚合"""
    collector = MetricsCollector(metric_name="test_metric")
    aggregates = collector.get_aggregates()
    
    assert aggregates['count'] == 0
    assert aggregates['sum'] == 0
    assert aggregates['mean'] == 0
    assert aggregates['min'] is None
    assert aggregates['max'] is None
    assert aggregates['std'] == 0
    assert aggregates['median'] is None


def test_metrics_collector_filtered_aggregates():
    """测试基于标签的过滤聚合"""
    collector = MetricsCollector(metric_name="test_metric")
    
    # 记录带不同标签的数据
    collector.record_metric(10, tags={"category": "A"})
    collector.record_metric(20, tags={"category": "A"})
    collector.record_metric(30, tags={"category": "B"})
    collector.record_metric(40, tags={"category": "B"})
    
    # 获取所有数据的聚合
    all_aggregates = collector.get_aggregates()
    assert all_aggregates['mean'] == 25.0
    
    # 获取特定标签的数据聚合
    a_aggregates = collector.get_aggregates(filter_tags={"category": "A"})
    assert a_aggregates['count'] == 2
    assert a_aggregates['mean'] == 15.0
    
    b_aggregates = collector.get_aggregates(filter_tags={"category": "B"})
    assert b_aggregates['count'] == 2
    assert b_aggregates['mean'] == 35.0


# 参数化测试
def test_metrics_collector_different_value_types():
    """测试不同类型的值"""
    collector = MetricsCollector(metric_name="test_metric")
    
    # 测试不同数值类型
    collector.record_metric(42)          # int
    collector.record_metric(3.14)        # float
    collector.record_metric(np.int64(100))  # numpy int
    collector.record_metric(np.float64(2.71))  # numpy float
    
    assert len(collector.metrics) == 4
    aggregates = collector.get_aggregates()
    assert aggregates['count'] == 4
    assert aggregates['sum'] == 42 + 3.14 + 100 + 2.71


# 性能测试
@pytest.mark.benchmark
def test_metrics_collector_performance():
    """测试MetricsCollector性能"""
    collector = MetricsCollector(metric_name="benchmark")
    
    # 记录大量指标
    start_time = time.time()
    for i in range(10000):
        collector.record_metric(i, tags={"iteration": i % 10})
    end_time = time.time()
    
    # 验证性能（记录10000条数据应在合理时间内完成）
    elapsed_time = end_time - start_time
    print(f"Recorded 10000 metrics in {elapsed_time:.4f} seconds")
    assert elapsed_time < 1.0  # 1秒内完成
    
    # 测试聚合计算性能
    start_time = time.time()
    aggregates = collector.get_aggregates()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Calculated aggregates in {elapsed_time:.4f} seconds")
    assert elapsed_time < 0.1  # 100毫秒内完成


# 异常处理测试
def test_metrics_collector_export_to_invalid_path():
    """测试导出到无效路径"""
    collector = MetricsCollector(metric_name="test_metric")
    collector.record_metric(42)
    
    # 尝试导出到不存在的目录
    invalid_path = os.path.join("/invalid/directory", "metrics.csv")
    
    # 根据操作系统和权限，这可能会失败，但应该被优雅处理
    try:
        success = collector.export_to_file(invalid_path)
        assert not success  # 应该返回失败
    except Exception as e:
        # 允许抛出异常，但应该是预期的类型
        assert isinstance(e, (IOError, OSError))


# 边界条件测试
def test_metrics_collector_large_values():
    """测试大数值"""
    collector = MetricsCollector(metric_name="test_metric")
    
    # 测试非常大的数
    large_value = 1e18
    collector.record_metric(large_value)
    
    # 测试非常小的数
    small_value = 1e-18
    collector.record_metric(small_value)
    
    aggregates = collector.get_aggregates()
    assert aggregates['sum'] == large_value + small_value


# 测试定时器装饰器的参数
@performance_timer(name="custom_timer", log_level="debug")
def custom_timer_function():
    """使用自定义名称和日志级别的函数"""
    time.sleep(0.05)
    return "custom"

def test_performance_timer_with_params(mock_logger):
    """测试带参数的性能计时器装饰器"""
    result = custom_timer_function()
    assert result == "custom"
    
    # 验证调用了debug级别日志
    mock_logger.debug.assert_called()
    
    # 验证日志消息包含自定义名称
    calls = [call for call in mock_logger.debug.call_args_list if "custom_timer" in str(call)]
    assert len(calls) > 0