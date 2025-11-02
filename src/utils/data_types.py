#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据类型模块 (Data Types)
定义统一的数据类型和数据结构
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid
import numpy as np
from enum import Enum, auto


class SimulationMode(Enum):
    """
    模拟模式枚举
    """
    REALISTIC = auto()  # 现实主义模式
    OPTIMISTIC = auto()  # 乐观主义模式
    PESSIMISTIC = auto()  # 悲观主义模式
    RANDOM = auto()      # 随机模式
    CUSTOM = auto()      # 自定义模式


class DomainType(Enum):
    """
    领域类型枚举
    """
    ECONOMIC = auto()     # 经济领域
    POLITICAL = auto()    # 政治领域
    TECHNOLOGICAL = auto()  # 技术领域
    CLIMATE = auto()      # 气候领域
    SOCIAL = auto()       # 社会领域
    HEALTH = auto()       # 健康领域
    ENVIRONMENTAL = auto()  # 环境领域
    ENERGY = auto()       # 能源领域
    DEFENSE = auto()      # 国防领域
    SPACE = auto()        # 太空领域


class EventType(Enum):
    """
    事件类型枚举
    """
    CRITICAL = auto()     # 关键事件
    CATACLYSMIC = auto()  # 灾难性事件
    OPPORTUNITY = auto()  # 机会事件
    NORMAL = auto()       # 普通事件
    TIPPING_POINT = auto()  # 转折点事件
    BLACK_SWAN = auto()   # 黑天鹅事件


class InsightType(Enum):
    """
    洞察类型枚举
    """
    TREND = auto()          # 趋势洞察
    CYCLE = auto()          # 周期洞察
    RISK = auto()           # 风险洞察
    OPPORTUNITY = auto()    # 机会洞察
    THRESHOLD = auto()      # 阈值洞察
    CORRELATION = auto()    # 相关性洞察
    CASCADING = auto()      # 级联效应洞察
    BREAKING_POINT = auto()  # 突破点洞察


@dataclass(frozen=True)
class SimulationConfig:
    """
    模拟配置类
    """
    mode: SimulationMode = SimulationMode.REALISTIC
    time_step: int = 1  # 时间步长（年）
    max_steps: int = 50  # 最大时间步数
    initial_year: int = 2023  # 初始年份
    seed: Optional[int] = None  # 随机种子
    population_size: int = 1000  # 世界线初始数量
    survival_rate: float = 0.1  # 存活率
    domains: List[DomainType] = field(default_factory=lambda: [])  # 启用的领域
    convergence_threshold: float = 0.01  # 收敛阈值
    enable_nonlinearity: bool = True  # 是否启用非线性
    enable_cascading_effects: bool = True  # 是否启用级联效应
    enable_sensitivity_analysis: bool = False  # 是否启用敏感性分析


@dataclass
class DomainState:
    """
    领域状态类
    """
    domain_type: DomainType
    metrics: Dict[str, float] = field(default_factory=dict)  # 指标名称 -> 指标值
    trends: Dict[str, float] = field(default_factory=dict)  # 指标名称 -> 趋势值
    volatility: Dict[str, float] = field(default_factory=dict)  # 指标名称 -> 波动率
    tipping_points: Dict[str, float] = field(default_factory=dict)  # 指标名称 -> 距离临界点的距离
    dependencies: List[DomainType] = field(default_factory=list)  # 依赖的领域
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def __post_init__(self):
        # 确保所有字典都被正确初始化
        self.metrics = self.metrics or {}
        self.trends = self.trends or {}
        self.volatility = self.volatility or {}
        self.tipping_points = self.tipping_points or {}
        self.dependencies = self.dependencies or []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'domain_type': self.domain_type.name,
            'metrics': self.metrics,
            'trends': self.trends,
            'volatility': self.volatility,
            'tipping_points': self.tipping_points,
            'dependencies': [d.name for d in self.dependencies],
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainState':
        """从字典创建实例"""
        return cls(
            domain_type=DomainType[data['domain_type']],
            metrics=data.get('metrics', {}),
            trends=data.get('trends', {}),
            volatility=data.get('volatility', {}),
            tipping_points=data.get('tipping_points', {}),
            dependencies=[DomainType[d] for d in data.get('dependencies', [])],
            timestamp=data.get('timestamp', datetime.now().timestamp())
        )


@dataclass
class WorldEvent:
    """
    世界事件类
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    event_type: EventType = EventType.NORMAL
    domains: List[DomainType] = field(default_factory=list)
    severity: float = 0.5  # 严重程度 0-1
    probability: float = 1.0  # 概率 0-1
    impact: Dict[DomainType, Dict[str, float]] = field(default_factory=dict)  # 领域 -> {指标 -> 影响值}
    timestamp: int = 0  # 发生时间（模拟年份）
    location: Optional[str] = None  # 地理位置
    cascading_events: List[str] = field(default_factory=list)  # 级联事件ID列表
    
    def __post_init__(self):
        # 确保所有字段都被正确初始化
        self.domains = self.domains or []
        self.impact = self.impact or {}
        self.cascading_events = self.cascading_events or []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_id': self.event_id,
            'name': self.name,
            'description': self.description,
            'event_type': self.event_type.name,
            'domains': [d.name for d in self.domains],
            'severity': self.severity,
            'probability': self.probability,
            'impact': {d.name: v for d, v in self.impact.items()},
            'timestamp': self.timestamp,
            'location': self.location,
            'cascading_events': self.cascading_events
        }


@dataclass
class WorldLine:
    """
    世界线类
    """
    world_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    current_year: int = 2023
    domain_states: Dict[DomainType, DomainState] = field(default_factory=dict)
    history: List[Dict[DomainType, DomainState]] = field(default_factory=list)
    events: List[WorldEvent] = field(default_factory=list)
    seed: Optional[int] = None  # 生成此世界线的种子
    probability_score: float = 1.0  # 概率分数
    reality_score: float = 0.0  # 现实符合度分数
    survival_status: bool = True  # 是否存活
    tags: Dict[str, Any] = field(default_factory=dict)  # 标签
    
    def __post_init__(self):
        # 确保所有字段都被正确初始化
        self.domain_states = self.domain_states or {}
        self.history = self.history or []
        self.events = self.events or []
        self.tags = self.tags or {}
    
    def add_domain_state(self, domain_state: DomainState) -> None:
        """添加领域状态"""
        self.domain_states[domain_state.domain_type] = domain_state
    
    def record_history(self) -> None:
        """记录当前状态到历史"""
        # 深拷贝当前状态
        state_copy = {k: DomainState(**asdict(v)) for k, v in self.domain_states.items()}
        self.history.append({
            'year': self.current_year,
            'states': state_copy
        })
    
    def add_event(self, event: WorldEvent) -> None:
        """添加事件"""
        self.events.append(event)
    
    def get_domain_metric(self, domain_type: DomainType, metric_name: str) -> Optional[float]:
        """获取领域指标值"""
        if domain_type in self.domain_states:
            return self.domain_states[domain_type].metrics.get(metric_name)
        return None
    
    def update_probability_score(self, score: float) -> None:
        """更新概率分数"""
        self.probability_score = score
    
    def update_reality_score(self, score: float) -> None:
        """更新现实符合度分数"""
        self.reality_score = score
    
    def mark_survived(self, survived: bool) -> None:
        """标记存活状态"""
        self.survival_status = survived
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'world_id': self.world_id,
            'name': self.name,
            'description': self.description,
            'current_year': self.current_year,
            'domain_states': {d.name: s.to_dict() for d, s in self.domain_states.items()},
            'history': [
                {
                    'year': h['year'],
                    'states': {d.name: s.to_dict() for d, s in h['states'].items()}
                }
                for h in self.history
            ],
            'events': [e.to_dict() for e in self.events],
            'seed': self.seed,
            'probability_score': self.probability_score,
            'reality_score': self.reality_score,
            'survival_status': self.survival_status,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorldLine':
        """从字典创建实例"""
        world_line = cls(
            world_id=data['world_id'],
            name=data.get('name', ''),
            description=data.get('description', ''),
            current_year=data.get('current_year', 2023),
            seed=data.get('seed'),
            probability_score=data.get('probability_score', 1.0),
            reality_score=data.get('reality_score', 0.0),
            survival_status=data.get('survival_status', True),
            tags=data.get('tags', {})
        )
        
        # 恢复领域状态
        for domain_name, state_data in data.get('domain_states', {}).items():
            domain_type = DomainType[domain_name]
            state = DomainState.from_dict(state_data)
            world_line.add_domain_state(state)
        
        # 恢复历史
        for h_data in data.get('history', []):
            history_entry = {
                'year': h_data['year'],
                'states': {}
            }
            for domain_name, state_data in h_data['states'].items():
                domain_type = DomainType[domain_name]
                state = DomainState.from_dict(state_data)
                history_entry['states'][domain_type] = state
            world_line.history.append(history_entry)
        
        # 恢复事件（简化版）
        for event_data in data.get('events', []):
            event = WorldEvent(**event_data)
            world_line.events.append(event)
        
        return world_line


@dataclass
class RealityData:
    """
    现实数据类，用于存储和处理现实世界的参考数据
    """
    data_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    source: str = ""  # 数据源
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    data_points: Dict[Tuple[DomainType, str], List[Tuple[int, float]]] = field(default_factory=dict)  # ((领域, 指标), [(年份, 值)])
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def __post_init__(self):
        self.data_points = self.data_points or {}
        self.metadata = self.metadata or {}
    
    def add_data_point(self, domain_type: DomainType, metric_name: str,
                      year: int, value: float) -> None:
        """添加数据点"""
        key = (domain_type, metric_name)
        if key not in self.data_points:
            self.data_points[key] = []
        self.data_points[key].append((year, value))
        
        # 按年份排序
        self.data_points[key].sort(key=lambda x: x[0])
    
    def get_data_points(self, domain_type: DomainType, metric_name: str) -> List[Tuple[int, float]]:
        """获取指定领域和指标的数据点"""
        key = (domain_type, metric_name)
        return self.data_points.get(key, [])
    
    def get_value_by_year(self, domain_type: DomainType, metric_name: str,
                         year: int) -> Optional[float]:
        """获取指定年份的值"""
        data_points = self.get_data_points(domain_type, metric_name)
        for y, v in data_points:
            if y == year:
                return v
        return None
    
    def get_latest_value(self, domain_type: DomainType, metric_name: str) -> Optional[float]:
        """获取最新值"""
        data_points = self.get_data_points(domain_type, metric_name)
        if data_points:
            return data_points[-1][1]
        return None
    
    def get_available_metrics(self, domain_type: DomainType) -> List[str]:
        """获取指定领域可用的指标列表"""
        metrics = []
        for (d, m), _ in self.data_points.items():
            if d == domain_type and m not in metrics:
                metrics.append(m)
        return metrics
    
    def get_available_domains(self) -> List[DomainType]:
        """获取可用的领域列表"""
        domains = set()
        for (d, _), _ in self.data_points.items():
            domains.add(d)
        return list(domains)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_id': self.data_id,
            'name': self.name,
            'source': self.source,
            'timestamp': self.timestamp,
            'data_points': {
                f"{domain_type.name}:{metric_name}": points
                for (domain_type, metric_name), points in self.data_points.items()
            },
            'metadata': self.metadata
        }


@dataclass
class Insight:
    """
    洞察类
    """
    insight_type: InsightType  # 必需参数放在最前面
    insight_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 标识符
    title: str = ""
    description: str = ""
    domains: List[DomainType] = field(default_factory=list)
    confidence: float = 0.5  # 置信度 0-1
    severity: float = 0.5  # 严重程度 0-1
    time_range: Tuple[int, int] = (0, 0)  # 时间范围 (起始年, 结束年)
    related_worldlines: List[str] = field(default_factory=list)  # 相关的世界线ID
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)  # 支持证据
    recommendations: List[str] = field(default_factory=list)  # 建议
    source: str = ""  # 来源
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def __post_init__(self):
        self.domains = self.domains or []
        self.related_worldlines = self.related_worldlines or []
        self.supporting_evidence = self.supporting_evidence or []
        self.recommendations = self.recommendations or []
    
    def add_evidence(self, evidence: Dict[str, Any]) -> None:
        """添加支持证据"""
        self.supporting_evidence.append(evidence)
    
    def add_recommendation(self, recommendation: str) -> None:
        """添加建议"""
        self.recommendations.append(recommendation)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'insight_id': self.insight_id,
            'insight_type': self.insight_type.name,
            'title': self.title,
            'description': self.description,
            'domains': [d.name for d in self.domains],
            'confidence': self.confidence,
            'severity': self.severity,
            'time_range': self.time_range,
            'related_worldlines': self.related_worldlines,
            'supporting_evidence': self.supporting_evidence,
            'recommendations': self.recommendations,
            'source': self.source,
            'timestamp': self.timestamp
        }


@dataclass
class SimulationResult:
    """
    模拟结果类
    """
    simulation_config: SimulationConfig  # 必需参数放在前面
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 默认参数放在后面
    worldlines: List[WorldLine] = field(default_factory=list)
    surviving_worldlines: List[WorldLine] = field(default_factory=list)
    insights: List[Insight] = field(default_factory=list)
    start_time: float = field(default_factory=lambda: datetime.now().timestamp())
    end_time: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)  # 模拟性能指标
    summary: Dict[str, Any] = field(default_factory=dict)  # 模拟总结
    
    def __post_init__(self):
        self.worldlines = self.worldlines or []
        self.surviving_worldlines = self.surviving_worldlines or []
        self.insights = self.insights or []
        self.metrics = self.metrics or {}
        self.summary = self.summary or {}
    
    def add_worldline(self, worldline: WorldLine) -> None:
        """添加世界线"""
        self.worldlines.append(worldline)
        if worldline.survival_status:
            self.surviving_worldlines.append(worldline)
    
    def add_insight(self, insight: Insight) -> None:
        """添加洞察"""
        self.insights.append(insight)
    
    def finalize(self) -> None:
        """完成模拟结果"""
        self.end_time = datetime.now().timestamp()
        self._update_summary()
    
    def _update_summary(self) -> None:
        """更新总结信息"""
        total_worldlines = len(self.worldlines)
        surviving_count = len(self.surviving_worldlines)
        survival_rate = surviving_count / total_worldlines if total_worldlines > 0 else 0
        
        # 计算平均概率分数
        avg_probability = np.mean([w.probability_score for w in self.surviving_worldlines]) if surviving_count > 0 else 0
        
        # 计算平均现实符合度
        avg_reality = np.mean([w.reality_score for w in self.surviving_worldlines]) if surviving_count > 0 else 0
        
        # 收集所有事件
        all_events = []
        for worldline in self.worldlines:
            all_events.extend(worldline.events)
        
        # 按类型统计事件
        event_counts = {}
        for event in all_events:
            event_type = event.event_type.name
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        self.summary = {
            'total_worldlines': total_worldlines,
            'surviving_worldlines': surviving_count,
            'survival_rate': survival_rate,
            'avg_probability_score': avg_probability,
            'avg_reality_score': avg_reality,
            'total_events': len(all_events),
            'events_by_type': event_counts,
            'total_insights': len(self.insights),
            'simulation_duration': self.end_time - self.start_time,
            'final_year': max([w.current_year for w in self.worldlines]) if self.worldlines else self.simulation_config.initial_year
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'result_id': self.result_id,
            'simulation_config': asdict(self.simulation_config),
            'worldlines': [w.to_dict() for w in self.worldlines[:100]],  # 限制序列化的世界线数量
            'surviving_worldlines': [w.to_dict() for w in self.surviving_worldlines[:100]],
            'insights': [i.to_dict() for i in self.insights],
            'start_time': self.start_time,
            'end_time': self.end_time,
            'metrics': self.metrics,
            'summary': self.summary
        }


@dataclass
class ImpactMatrix:
    """
    领域影响矩阵类
    """
    matrix: Dict[Tuple[DomainType, DomainType], float] = field(default_factory=dict)
    domains: List[DomainType] = field(default_factory=list)
    
    def __post_init__(self):
        self.matrix = self.matrix or {}
        self.domains = self.domains or []
    
    def set_impact(self, source: DomainType, target: DomainType, impact_value: float) -> None:
        """设置领域间的影响值"""
        if source not in self.domains:
            self.domains.append(source)
        if target not in self.domains:
            self.domains.append(target)
        self.matrix[(source, target)] = impact_value
    
    def get_impact(self, source: DomainType, target: DomainType) -> float:
        """获取领域间的影响值"""
        return self.matrix.get((source, target), 0.0)
    
    def get_impacts_from(self, source: DomainType) -> Dict[DomainType, float]:
        """获取从指定领域出发的所有影响"""
        impacts = {}
        for (s, t), v in self.matrix.items():
            if s == source:
                impacts[t] = v
        return impacts
    
    def get_impacts_to(self, target: DomainType) -> Dict[DomainType, float]:
        """获取作用于指定领域的所有影响"""
        impacts = {}
        for (s, t), v in self.matrix.items():
            if t == target:
                impacts[s] = v
        return impacts
    
    def normalize(self) -> None:
        """标准化影响矩阵"""
        max_impact = max(abs(v) for v in self.matrix.values()) if self.matrix else 1.0
        if max_impact > 0:
            for key in self.matrix:
                self.matrix[key] = self.matrix[key] / max_impact
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'matrix': {
                f"{source.name}->{target.name}": value
                for (source, target), value in self.matrix.items()
            },
            'domains': [d.name for d in self.domains]
        }


@dataclass
class SensitivityAnalysisResult:
    """
    敏感性分析结果类
    """
    parameter_name: str
    parameter_values: List[float]
    output_metrics: Dict[str, List[float]] = field(default_factory=dict)
    sensitivity_index: float = 0.0  # 敏感性指数
    correlation_coefficient: float = 0.0  # 相关系数
    
    def __post_init__(self):
        self.output_metrics = self.output_metrics or {}
    
    def add_metric(self, metric_name: str, values: List[float]) -> None:
        """添加指标值列表"""
        self.output_metrics[metric_name] = values
    
    def calculate_correlation(self, metric_name: str) -> float:
        """计算相关系数"""
        if metric_name not in self.output_metrics:
            return 0.0
        
        try:
            from scipy.stats import pearsonr
            corr, _ = pearsonr(self.parameter_values, self.output_metrics[metric_name])
            self.correlation_coefficient = corr
            return corr
        except ImportError:
            # 如果没有scipy，使用numpy的简化计算
            x = np.array(self.parameter_values)
            y = np.array(self.output_metrics[metric_name])
            corr = np.corrcoef(x, y)[0, 1]
            self.correlation_coefficient = corr
            return corr
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'parameter_name': self.parameter_name,
            'parameter_values': self.parameter_values,
            'output_metrics': self.output_metrics,
            'sensitivity_index': self.sensitivity_index,
            'correlation_coefficient': self.correlation_coefficient
        }