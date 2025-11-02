#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据类型模块 (Data Types)
定义数据模块中使用的核心数据类型
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import uuid

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_types import DomainType, SimulationMode


class DataSourceType(Enum):
    """
    数据源类型枚举
    """
    WORLD_BANK = "world_bank"
    IMF = "imf"
    NASA = "nasa"
    UN = "united_nations"
    GITHUB = "github"
    CUSTOM_API = "custom_api"
    CSV_FILE = "csv_file"
    DATABASE = "database"
    SIMULATION = "simulation"
    WEB_SCRAPE = "web_scrape"
    OTHER = "other"


class DataFormat(Enum):
    """
    数据格式枚举
    """
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    XML = "xml"
    SQL = "sql"
    PARQUET = "parquet"
    PICKLE = "pickle"
    TEXT = "text"
    BINARY = "binary"


class DataQualityLevel(Enum):
    """
    数据质量级别枚举
    """
    RAW = "raw"  # 原始数据
    CLEANED = "cleaned"  # 清洗后的数据
    PROCESSED = "processed"  # 处理后的数据
    FEATURED = "featured"  # 特征工程后的数据
    ANALYZED = "analyzed"  # 分析后的数据


@dataclass
class DataPoint:
    """
    单个数据点
    """
    year: int  # 年份
    value: float  # 值
    confidence: float = 1.0  # 置信度（0-1）
    source: Optional[str] = None  # 来源
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataPoint':
        """从字典创建"""
        return cls(**data)
    
    def is_valid(self) -> bool:
        """检查数据点是否有效"""
        return (isinstance(self.year, int) and 
                isinstance(self.value, (int, float)) and 
                0 <= self.confidence <= 1)


@dataclass
class MetricData:
    """
    指标数据集合
    """
    metric_name: str  # 指标名称
    domain_type: DomainType  # 领域类型
    description: str = ""  # 指标描述
    unit: str = ""  # 单位
    data_points: List[DataPoint] = field(default_factory=list)  # 数据点列表
    last_updated: datetime = field(default_factory=datetime.now)  # 最后更新时间
    quality_level: DataQualityLevel = DataQualityLevel.RAW  # 质量级别
    
    def __post_init__(self):
        # 确保数据点按年份排序
        self.data_points.sort(key=lambda x: x.year)
    
    def add_data_point(self, year: int, value: float, confidence: float = 1.0,
                      source: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加数据点
        
        Args:
            year: 年份
            value: 值
            confidence: 置信度
            source: 来源
            metadata: 元数据
        """
        # 检查是否已存在该年份的数据点
        existing_index = None
        for i, dp in enumerate(self.data_points):
            if dp.year == year:
                existing_index = i
                break
        
        # 创建新数据点
        new_dp = DataPoint(
            year=year,
            value=value,
            confidence=confidence,
            source=source,
            metadata=metadata or {}
        )
        
        # 如果存在，替换；否则添加
        if existing_index is not None:
            self.data_points[existing_index] = new_dp
        else:
            self.data_points.append(new_dp)
        
        # 重新排序
        self.data_points.sort(key=lambda x: x.year)
        self.last_updated = datetime.now()
    
    def get_value_for_year(self, year: int) -> Optional[float]:
        """
        获取指定年份的值
        
        Args:
            year: 年份
            
        Returns:
            值，如果不存在返回None
        """
        for dp in self.data_points:
            if dp.year == year:
                return dp.value
        return None
    
    def get_range(self) -> Tuple[int, int]:
        """
        获取数据的年份范围
        
        Returns:
            (最小年份, 最大年份)
        """
        if not self.data_points:
            return (0, 0)
        return (min(dp.year for dp in self.data_points),
                max(dp.year for dp in self.data_points))
    
    def get_stats(self) -> Dict[str, float]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        if not self.data_points:
            return {}
        
        values = [dp.value for dp in self.data_points]
        return {
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'count': len(values)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'metric_name': self.metric_name,
            'domain_type': self.domain_type.value,
            'description': self.description,
            'unit': self.unit,
            'data_points': [dp.to_dict() for dp in self.data_points],
            'last_updated': self.last_updated.isoformat(),
            'quality_level': self.quality_level.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricData':
        """从字典创建"""
        return cls(
            metric_name=data['metric_name'],
            domain_type=DomainType(data['domain_type']),
            description=data.get('description', ''),
            unit=data.get('unit', ''),
            data_points=[DataPoint.from_dict(dp) for dp in data.get('data_points', [])],
            last_updated=datetime.fromisoformat(data.get('last_updated', datetime.now().isoformat())),
            quality_level=DataQualityLevel(data.get('quality_level', 'raw'))
        )


@dataclass
class Dataset:
    """
    数据集基类
    """
    name: str  # 数据集名称
    source: str  # 数据源
    description: str = ""  # 数据集描述
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    created_at: datetime = field(default_factory=datetime.now)  # 创建时间
    updated_at: datetime = field(default_factory=datetime.now)  # 更新时间
    quality_level: DataQualityLevel = DataQualityLevel.RAW  # 质量级别
    
    def update_timestamp(self) -> None:
        """更新时间戳"""
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'source': self.source,
            'description': self.description,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'quality_level': self.quality_level.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dataset':
        """从字典创建"""
        return cls(
            name=data['name'],
            source=data['source'],
            description=data.get('description', ''),
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat())),
            quality_level=DataQualityLevel(data.get('quality_level', 'raw'))
        )


@dataclass
class RealityData(Dataset):
    """
    现实数据类
    存储从各种来源收集的现实世界数据
    """
    data_points: Dict[Tuple[DomainType, str], List[Tuple[int, float]]] = field(
        default_factory=dict
    )  # 数据点字典，键为(领域类型, 指标名称)
    
    def add_data_point(self, domain_type: DomainType, metric_name: str,
                      year: int, value: float) -> None:
        """
        添加数据点
        
        Args:
            domain_type: 领域类型
            metric_name: 指标名称
            year: 年份
            value: 值
        """
        key = (domain_type, metric_name)
        if key not in self.data_points:
            self.data_points[key] = []
        
        # 检查是否已存在该年份的数据
        existing_index = None
        for i, (y, _) in enumerate(self.data_points[key]):
            if y == year:
                existing_index = i
                break
        
        # 更新或添加数据点
        if existing_index is not None:
            self.data_points[key][existing_index] = (year, value)
        else:
            self.data_points[key].append((year, value))
        
        # 排序
        self.data_points[key].sort(key=lambda x: x[0])
        self.update_timestamp()
    
    def get_data_points(self, domain_type: DomainType,
                       metric_name: str) -> List[Tuple[int, float]]:
        """
        获取指定领域和指标的数据点
        
        Args:
            domain_type: 领域类型
            metric_name: 指标名称
            
        Returns:
            数据点列表
        """
        key = (domain_type, metric_name)
        return self.data_points.get(key, [])
    
    def get_metrics(self, domain_type: Optional[DomainType] = None) -> List[Tuple[DomainType, str]]:
        """
        获取所有指标
        
        Args:
            domain_type: 可选的领域类型过滤
            
        Returns:
            指标列表
        """
        if domain_type is None:
            return list(self.data_points.keys())
        
        return [(dt, mn) for dt, mn in self.data_points.keys() if dt == domain_type]
    
    def get_value(self, domain_type: DomainType, metric_name: str,
                 year: int) -> Optional[float]:
        """
        获取指定年份的值
        
        Args:
            domain_type: 领域类型
            metric_name: 指标名称
            year: 年份
            
        Returns:
            值，如果不存在返回None
        """
        data_points = self.get_data_points(domain_type, metric_name)
        for y, v in data_points:
            if y == year:
                return v
        return None
    
    def get_range(self) -> Tuple[int, int]:
        """
        获取数据的年份范围
        
        Returns:
            (最小年份, 最大年份)
        """
        all_years = []
        for points in self.data_points.values():
            for year, _ in points:
                all_years.append(year)
        
        if not all_years:
            return (0, 0)
        
        return (min(all_years), max(all_years))
    
    def get_domains(self) -> Set[DomainType]:
        """
        获取所有包含的领域
        
        Returns:
            领域集合
        """
        return {domain_type for domain_type, _ in self.data_points.keys()}
    
    def merge(self, other: 'RealityData') -> 'RealityData':
        """
        合并另一个RealityData
        
        Args:
            other: 另一个RealityData实例
            
        Returns:
            合并后的RealityData
        """
        merged = RealityData(
            name=f"merged_{self.name}_{other.name}",
            source=f"merged:{self.source};{other.source}",
            description=f"Merged data from {self.name} and {other.name}",
            metadata={
                'sources': [self.source, other.source],
                'merged_at': datetime.now().isoformat()
            }
        )
        
        # 添加当前数据
        for (domain_type, metric_name), points in self.data_points.items():
            for year, value in points:
                merged.add_data_point(domain_type, metric_name, year, value)
        
        # 添加其他数据
        for (domain_type, metric_name), points in other.data_points.items():
            for year, value in points:
                merged.add_data_point(domain_type, metric_name, year, value)
        
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base_dict = super().to_dict()
        # 转换领域类型为字符串以便序列化
        serializable_data_points = {}
        for (domain_type, metric_name), points in self.data_points.items():
            key = f"{domain_type.value}_{metric_name}"
            serializable_data_points[key] = points
        
        base_dict['data_points'] = serializable_data_points
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RealityData':
        """从字典创建"""
        instance = super().from_dict(data)
        
        # 反序列化数据点
        for key, points in data.get('data_points', {}).items():
            # 解析领域类型和指标名称
            parts = key.split('_', 1)
            if len(parts) == 2:
                domain_type_str, metric_name = parts
                try:
                    domain_type = DomainType(domain_type_str)
                    for year, value in points:
                        instance.add_data_point(domain_type, metric_name, year, value)
                except ValueError:
                    # 忽略无效的领域类型
                    continue
        
        return instance


@dataclass
class WorldLine:
    """
    世界线类
    表示一个可能的未来世界演化路径
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 世界线ID
    seed: Optional[int] = None  # 随机种子
    generation: int = 0  # 世代
    birth_time: str = field(default_factory=lambda: datetime.now().isoformat())  # 出生时间
    parent_ids: List[str] = field(default_factory=list)  # 父世界线ID列表
    survival_score: float = 0.0  # 生存评分
    survival_probability: float = 1.0  # 生存概率（测试需要）
    value_score: float = 0.0  # 价值评分（测试需要）
    current_time: str = field(default_factory=lambda: datetime.now().isoformat())  # 当前时间
    state: Dict[str, Any] = field(default_factory=dict)  # 状态
    events: List[Dict[str, Any]] = field(default_factory=list)  # 事件列表
    history: List[Dict[str, Any]] = field(default_factory=list)  # 历史记录
    # 测试文件中使用的参数
    initial_state: Dict[str, Any] = field(default_factory=dict)  # 初始状态
    timeline: List[Dict[str, Any]] = field(default_factory=list)  # 时间线
    
    def copy(self) -> 'WorldLine':
        """
        创建世界线的深拷贝
        
        Returns:
            深拷贝的世界线
        """
        import copy
        return copy.deepcopy(self)
    

    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'seed': self.seed,
            'generation': self.generation,
            'birth_time': self.birth_time,
            'parent_ids': self.parent_ids,
            'survival_score': self.survival_score,
            'current_time': self.current_time,
            'state': self.state,
            'events': self.events,
            'history': self.history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorldLine':
        """
        从字典创建
        """
        return cls(
            id=data['id'],
            seed=data.get('seed'),
            generation=data.get('generation', 0),
            birth_time=data.get('birth_time', datetime.now().isoformat()),
            parent_ids=data.get('parent_ids', []),
            survival_score=data.get('survival_score', 0.0),
            current_time=data.get('current_time', datetime.now().isoformat()),
            state=data.get('state', {}),
            events=data.get('events', []),
            history=data.get('history', [])
        )


@dataclass
class SimulationResult:
    """
    模拟结果类
    """
    simulation_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 模拟ID
    start_year: int = field(default_factory=lambda: datetime.now().year)  # 开始年份
    end_year: int = field(default_factory=lambda: datetime.now().year + 10)  # 结束年份
    worldlines: List[WorldLine] = field(default_factory=list)  # 生成的世界线列表
    surviving_worldlines: List[WorldLine] = field(default_factory=list)  # 存活的世界线列表
    execution_time: float = 0.0  # 执行时间（秒）
    parameters: Dict[str, Any] = field(default_factory=dict)  # 模拟参数
    metrics: Dict[str, float] = field(default_factory=dict)  # 模拟指标
    errors: List[Dict[str, Any]] = field(default_factory=list)  # 错误记录
    created_at: datetime = field(default_factory=datetime.now)  # 创建时间
    
    def add_worldline(self, worldline: WorldLine) -> None:
        """
        添加世界线
        
        Args:
            worldline: 世界线
        """
        self.worldlines.append(worldline)
    
    def add_surviving_worldline(self, worldline: WorldLine) -> None:
        """
        添加存活的世界线
        
        Args:
            worldline: 世界线
        """
        self.surviving_worldlines.append(worldline)
    
    def calculate_statistics(self) -> None:
        """
        计算模拟统计信息
        """
        # 计算存活率
        if self.worldlines:
            self.metrics['survival_rate'] = len(self.surviving_worldlines) / len(self.worldlines)
        
        # 计算平均分歧度
        if self.surviving_worldlines:
            avg_divergence = sum(w.divergence_score for w in self.surviving_worldlines) / len(self.surviving_worldlines)
            self.metrics['average_divergence'] = avg_divergence
        
        # 计算平均生存概率
        if self.surviving_worldlines:
            avg_survival_prob = sum(w.survival_probability for w in self.surviving_worldlines) / len(self.surviving_worldlines)
            self.metrics['average_survival_probability'] = avg_survival_prob
    
    def get_most_likely_worldline(self) -> Optional[WorldLine]:
        """
        获取最有可能的世界线（生存概率最高）
        
        Returns:
            最有可能的世界线
        """
        if not self.surviving_worldlines:
            return None
        
        return max(self.surviving_worldlines, key=lambda w: w.survival_probability)
    
    def get_worldline_by_id(self, world_id: str) -> Optional[WorldLine]:
        """
        根据ID获取世界线
        
        Args:
            world_id: 世界线ID
            
        Returns:
            世界线
        """
        for w in self.worldlines:
            if w.world_id == world_id:
                return w
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        """
        return {
            'simulation_id': self.simulation_id,
            'start_year': self.start_year,
            'end_year': self.end_year,
            'worldlines': [w.to_dict() for w in self.worldlines],
            'surviving_worldlines': [w.to_dict() for w in self.surviving_worldlines],
            'execution_time': self.execution_time,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'errors': self.errors,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationResult':
        """
        从字典创建
        """
        return cls(
            simulation_id=data['simulation_id'],
            start_year=data['start_year'],
            end_year=data['end_year'],
            worldlines=[WorldLine.from_dict(w) for w in data.get('worldlines', [])],
            surviving_worldlines=[WorldLine.from_dict(w) for w in data.get('surviving_worldlines', [])],
            execution_time=data.get('execution_time', 0.0),
            parameters=data.get('parameters', {}),
            metrics=data.get('metrics', {}),
            errors=data.get('errors', []),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
        )


@dataclass
class CacheEntry:
    """
    缓存条目类
    """
    key: str  # 缓存键
    data: Any  # 缓存数据
    created_at: datetime = field(default_factory=datetime.now)  # 创建时间
    expires_at: Optional[datetime] = None  # 过期时间
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def is_expired(self) -> bool:
        """
        检查是否过期
        
        Returns:
            是否过期
        """
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def get_age(self) -> timedelta:
        """
        获取缓存年龄
        
        Returns:
            时间差
        """
        return datetime.now() - self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        """
        return {
            'key': self.key,
            'data': self.data,  # 注意：可能需要自定义序列化
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """
        从字典创建
        """
        expires_at = data.get('expires_at')
        if expires_at:
            expires_at = datetime.fromisoformat(expires_at)
        
        return cls(
            key=data['key'],
            data=data['data'],
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            expires_at=expires_at,
            metadata=data.get('metadata', {})
        )


@dataclass
class DataSourceConfig:
    """
    数据源配置类
    """
    name: str  # 数据源名称
    type: DataSourceType  # 数据源类型
    url: Optional[str] = None  # API URL或文件路径
    api_key: Optional[str] = None  # API密钥
    username: Optional[str] = None  # 用户名
    password: Optional[str] = None  # 密码
    params: Dict[str, Any] = field(default_factory=dict)  # 其他参数
    refresh_interval: int = 86400  # 刷新间隔（秒）
    enabled: bool = True  # 是否启用
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        获取认证头
        
        Returns:
            认证头字典
        """
        headers = {}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
        return headers
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典（不包含敏感信息）
        """
        return {
            'name': self.name,
            'type': self.type.value,
            'url': self.url,
            'params': self.params,
            'refresh_interval': self.refresh_interval,
            'enabled': self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSourceConfig':
        """
        从字典创建
        """
        return cls(
            name=data['name'],
            type=DataSourceType(data['type']),
            url=data.get('url'),
            api_key=data.get('api_key'),
            username=data.get('username'),
            password=data.get('password'),
            params=data.get('params', {}),
            refresh_interval=data.get('refresh_interval', 86400),
            enabled=data.get('enabled', True)
        )


@dataclass
class DataCatalog:
    """
    数据目录类
    管理所有可用的数据资源
    """
    datasets: Dict[str, Dataset] = field(default_factory=dict)  # 数据集字典
    sources: Dict[str, DataSourceConfig] = field(default_factory=dict)  # 数据源配置字典
    last_updated: datetime = field(default_factory=datetime.now)  # 最后更新时间
    
    def add_dataset(self, dataset: Dataset) -> None:
        """
        添加数据集
        
        Args:
            dataset: 数据集
        """
        self.datasets[dataset.name] = dataset
        self.last_updated = datetime.now()
    
    def remove_dataset(self, dataset_name: str) -> bool:
        """
        移除数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            是否成功移除
        """
        if dataset_name in self.datasets:
            del self.datasets[dataset_name]
            self.last_updated = datetime.now()
            return True
        return False
    
    def get_dataset(self, dataset_name: str) -> Optional[Dataset]:
        """
        获取数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            数据集
        """
        return self.datasets.get(dataset_name)
    
    def add_source(self, source_config: DataSourceConfig) -> None:
        """
        添加数据源配置
        
        Args:
            source_config: 数据源配置
        """
        self.sources[source_config.name] = source_config
        self.last_updated = datetime.now()
    
    def get_source(self, source_name: str) -> Optional[DataSourceConfig]:
        """
        获取数据源配置
        
        Args:
            source_name: 数据源名称
            
        Returns:
            数据源配置
        """
        return self.sources.get(source_name)
    
    def get_enabled_sources(self) -> List[DataSourceConfig]:
        """
        获取所有启用的数据源
        
        Returns:
            数据源配置列表
        """
        return [s for s in self.sources.values() if s.enabled]
    
    def search_datasets(self, keyword: str) -> List[Dataset]:
        """
        搜索数据集
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            匹配的数据集列表
        """
        keyword_lower = keyword.lower()
        results = []
        
        for dataset in self.datasets.values():
            if (keyword_lower in dataset.name.lower() or 
                keyword_lower in dataset.description.lower() or
                keyword_lower in dataset.source.lower()):
                results.append(dataset)
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        """
        return {
            'datasets': {name: dataset.to_dict() for name, dataset in self.datasets.items()},
            'sources': {name: source.to_dict() for name, source in self.sources.items()},
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataCatalog':
        """
        从字典创建
        """
        instance = cls()
        
        # 加载数据集
        for name, dataset_dict in data.get('datasets', {}).items():
            # 根据数据类型创建相应的数据集实例
            if 'data_points' in dataset_dict:
                instance.add_dataset(RealityData.from_dict(dataset_dict))
            else:
                instance.add_dataset(Dataset.from_dict(dataset_dict))
        
        # 加载数据源
        for name, source_dict in data.get('sources', {}).items():
            instance.add_source(DataSourceConfig.from_dict(source_dict))
        
        instance.last_updated = datetime.fromisoformat(data.get('last_updated', datetime.now().isoformat()))
        return instance


# 类型别名
MetricName = str
Year = int
Value = float
DataPoints = List[Tuple[Year, Value]]
MetricDict = Dict[Tuple[DomainType, MetricName], DataPoints]
WorldLineList = List[WorldLine]


# 便捷函数
def create_cache_entry(key: str, data: Any, ttl: Optional[int] = None,
                      **metadata) -> CacheEntry:
    """
    创建缓存条目
    
    Args:
        key: 缓存键
        data: 缓存数据
        ttl: 生存时间（秒）
        **metadata: 元数据
        
    Returns:
        缓存条目
    """
    expires_at = None
    if ttl is not None:
        expires_at = datetime.now() + timedelta(seconds=ttl)
    
    return CacheEntry(
        key=key,
        data=data,
        expires_at=expires_at,
        metadata=metadata
    )


def generate_worldline_id(prefix: str = "world") -> str:
    """
    生成世界线ID
    
    Args:
        prefix: 前缀
        
    Returns:
        世界线ID
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def format_data_point(year: int, value: float, confidence: float = 1.0) -> str:
    """
    格式化数据点为字符串
    
    Args:
        year: 年份
        value: 值
        confidence: 置信度
        
    Returns:
        格式化的字符串
    """
    return f"Year {year}: {value} (confidence: {confidence:.2f})"


# 示例用法
if __name__ == "__main__":
    # 创建一个RealityData示例
    reality_data = RealityData(
        name="global_economy",
        source="world_bank",
        description="Global economic indicators"
    )
    
    # 添加数据点
    reality_data.add_data_point(DomainType.ECONOMIC, "gdp", 2020, 84.5)
    reality_data.add_data_point(DomainType.ECONOMIC, "gdp", 2021, 93.8)
    reality_data.add_data_point(DomainType.ECONOMIC, "gdp", 2022, 101.6)
    
    # 打印数据
    print(f"Dataset: {reality_data.name}")
    print(f"Metrics: {reality_data.get_metrics()}")
    print(f"GDP (2021): {reality_data.get_value(DomainType.ECONOMIC, 'gdp', 2021)}")