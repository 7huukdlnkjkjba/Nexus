# Nexus 项目测试配置文件
# 定义测试夹具(fixtures)和通用测试配置

import os
import sys
import tempfile
import pytest
from unittest import mock

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 测试配置
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
TEST_CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')


@pytest.fixture(scope='session', autouse=True)
def setup_test_environment():
    """设置测试环境，创建必要的目录"""
    # 创建测试数据目录
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_CONFIG_DIR, exist_ok=True)
    
    # 设置环境变量
    os.environ['NEXUS_ENV'] = 'test'
    os.environ['LOG_LEVEL'] = 'ERROR'  # 减少测试过程中的日志输出
    
    yield
    
    # 清理测试环境（如需）
    # 注意：通常不建议在fixture中清理，以便保留测试结果进行调试


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def mock_config():
    """模拟配置对象"""
    return {
        'simulation': {
            'num_worldlines': 10,
            'time_horizon': 5,
            'random_seed': 42
        },
        'data': {
            'cache_enabled': False,
            'update_interval': 86400
        },
        'models': {
            'economy': {
                'enabled': True,
                'weight': 1.0
            },
            'politics': {
                'enabled': True,
                'weight': 1.0
            },
            'technology': {
                'enabled': True,
                'weight': 1.0
            },
            'climate': {
                'enabled': True,
                'weight': 1.0
            }
        }
    }


@pytest.fixture
def mock_data_collector():
    """模拟数据采集器"""
    with mock.patch('nexus.data.data_collector.DataCollector') as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.collect_data.return_value = {'test': 'data'}
        mock_instance.get_metadata.return_value = {'source': 'test', 'timestamp': '2023-01-01'}
        yield mock_instance


@pytest.fixture
def mock_domain_model():
    """模拟领域模型"""
    with mock.patch('nexus.models.base_domain.DomainModel') as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.initialize.return_value = None
        mock_instance.step.return_value = {'state': 'updated'}
        mock_instance.get_state.return_value = {'state': 'current'}
        yield mock_instance


@pytest.fixture
def sample_worldline():
    """提供样本世界线数据"""
    from nexus.data.data_types import WorldLine, DomainState, WorldEvent
    
    # 创建领域状态
    economy_state = DomainState(domain='economy', data={'gdp': 100, 'inflation': 2.0})
    politics_state = DomainState(domain='politics', data={'stability': 0.8})
    tech_state = DomainState(domain='technology', data={'innovation_index': 0.7})
    climate_state = DomainState(domain='climate', data={'temperature': 1.1})
    
    # 创建世界线
    worldline = WorldLine(
        id='test-worldline-1',
        seed=42,
        initial_state={
            'economy': economy_state,
            'politics': politics_state,
            'technology': tech_state,
            'climate': climate_state
        },
        timeline=[],
        survival_probability=1.0
    )
    
    # 添加一些时间点
    for t in range(1, 6):
        # 更新状态
        economy_state.data['gdp'] *= 1.03
        economy_state.data['inflation'] = 2.0 + t * 0.1
        
        # 添加事件
        event = WorldEvent(
            time=t,
            domain='economy',
            event_type='economic_growth',
            description=f'Year {t} growth',
            impact={'magnitude': 0.1 * t}
        )
        
        # 记录时间线
        worldline.timeline.append({
            'time': t,
            'states': {
                'economy': economy_state.copy(),
                'politics': politics_state.copy(),
                'technology': tech_state.copy(),
                'climate': climate_state.copy()
            },
            'events': [event]
        })
    
    return worldline


@pytest.fixture
def sample_reality_data():
    """提供样本现实数据"""
    from nexus.data.data_types import RealityData, DataPoint
    
    # 创建数据点
    data_points = [
        DataPoint(year=2020, domain='economy', metric='gdp', value=85.0, source='world_bank'),
        DataPoint(year=2021, domain='economy', metric='gdp', value=90.0, source='world_bank'),
        DataPoint(year=2022, domain='economy', metric='gdp', value=95.0, source='world_bank'),
        DataPoint(year=2020, domain='climate', metric='temperature', value=1.0, source='nasa'),
        DataPoint(year=2021, domain='climate', metric='temperature', value=1.05, source='nasa'),
        DataPoint(year=2022, domain='climate', metric='temperature', value=1.1, source='nasa')
    ]
    
    # 创建现实数据
    reality_data = RealityData(
        start_year=2020,
        end_year=2022,
        data_points=data_points
    )
    
    return reality_data


@pytest.fixture
def mock_simulation_engine():
    """模拟模拟引擎"""
    with mock.patch('nexus.engine.simulation_engine.SimulationEngine') as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.initialize.return_value = None
        mock_instance.run.return_value = {'success': True, 'num_completed': 10}
        mock_instance.get_results.return_value = {'worldlines': []}
        yield mock_instance


@pytest.fixture
def mock_insight_extractor():
    """模拟洞察提取器"""
    with mock.patch('nexus.engine.insight_extractor.InsightExtractor') as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.extract_patterns.return_value = [{'pattern': 'test'}]
        mock_instance.detect_turning_points.return_value = [{'time': 3}]
        mock_instance.extract_risk_signals.return_value = [{'risk': 'test'}]
        mock_instance.identify_opportunity_windows.return_value = [{'window': 'test'}]
        mock_instance.calculate_confidence.return_value = 0.95
        yield mock_instance


@pytest.fixture
def mock_metrics_collector():
    """模拟指标收集器"""
    with mock.patch('nexus.utils.metrics.MetricsCollector') as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.record_metric.return_value = None
        mock_instance.get_aggregates.return_value = {'avg': 0.5}
        mock_instance.export_to_file.return_value = True
        mock_instance.plot_metrics.return_value = None
        yield mock_instance


@pytest.fixture
def mock_cache_manager():
    """模拟缓存管理器"""
    with mock.patch('nexus.data.cache_manager.CacheManager') as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.get.return_value = None  # 默认缓存未命中
        mock_instance.set.return_value = True
        mock_instance.delete.return_value = True
        mock_instance.clear.return_value = True
        mock_instance.get_stats.return_value = {'hits': 0, 'misses': 1}
        
        # 允许测试中自定义缓存行为
        def side_effect(key):
            if key == 'cached_data':
                return {'cached': True}
            return None
        
        mock_instance.get.side_effect = side_effect
        yield mock_instance


@pytest.fixture
def mock_logger():
    """模拟日志记录器"""
    with mock.patch('nexus.utils.logger.get_logger') as mock_get_logger:
        mock_logger = mock.MagicMock()
        mock_get_logger.return_value = mock_logger
        yield mock_logger


@pytest.fixture
def mock_random_generator():
    """模拟随机数生成器"""
    with mock.patch('nexus.utils.random_utils.RandomGenerator') as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.random.return_value = 0.5
        mock_instance.randint.return_value = 42
        mock_instance.normal.return_value = 0.0
        mock_instance.choice.return_value = 'option1'
        mock_instance.shuffle.return_value = None
        yield mock_instance


# 性能测试标记
@pytest.mark.benchmark
def pytest_configure(config):
    """配置性能测试标记"""
    config.addinivalue_line(
        "markers", "benchmark: mark a test as a performance benchmark"
    )


# 跳过慢速测试的选项
def pytest_addoption(parser):
    """添加pytest命令行选项"""
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="skip slow tests"
    )
    parser.addoption(
        "--run-benchmarks", action="store_true", default=False, help="run benchmark tests"
    )


def pytest_collection_modifyitems(config, items):
    """根据命令行选项修改测试收集"""
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="--skip-slow option used")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    if not config.getoption("--run-benchmarks"):
        skip_benchmark = pytest.mark.skip(reason="need --run-benchmarks option to run")
        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_benchmark)


# 测试数据生成器
def generate_test_data(size=100):
    """生成测试数据"""
    import numpy as np
    from datetime import datetime, timedelta
    
    data = []
    start_date = datetime(2020, 1, 1)
    
    for i in range(size):
        date = start_date + timedelta(days=i)
        data.append({
            'date': date,
            'value1': np.random.normal(100, 10),
            'value2': np.random.normal(50, 5),
            'category': np.random.choice(['A', 'B', 'C', 'D'])
        })
    
    return data


# 测试辅助函数
def assert_dict_contains_subset(subset, full_dict, msg=None):
    """断言字典包含子集"""
    for key, value in subset.items():
        assert key in full_dict, f"Key {key} not found in dictionary"
        if isinstance(value, dict) and isinstance(full_dict[key], dict):
            assert_dict_contains_subset(value, full_dict[key], msg)
        else:
            assert full_dict[key] == value, f"Value for key {key} does not match: {full_dict[key]} != {value}"


def assert_approx_equal(value1, value2, tolerance=1e-6, msg=None):
    """断言两个值近似相等"""
    assert abs(value1 - value2) < tolerance, msg or f"Values not approximately equal: {value1} vs {value2}"


def assert_dataframe_equals(df1, df2, ignore_index=False, check_dtype=True):
    """断言两个DataFrame相等"""
    import pandas as pd
    pd.testing.assert_frame_equal(df1, df2, check_names=True, check_dtype=check_dtype, ignore_index=ignore_index)