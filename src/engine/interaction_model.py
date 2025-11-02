# Nexus 领域交互模型
# 负责处理经济、政治、技术、气候等领域间的相互影响

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
from scipy.optimize import fsolve

from ..utils.logger import get_logger
from ..utils.random_utils import RandomGenerator
from ..utils.data_types import DomainType, SimulationMode


@dataclass
class ImpactMatrix:
    """领域间影响矩阵定义"""
    # 影响矩阵：impact_matrix[source_domain][target_domain] = influence_strength
    matrix: Dict[str, Dict[str, float]]
    # 影响类型：positive, negative, neutral
    impact_types: Dict[str, Dict[str, str]]
    # 影响延迟：影响生效所需的时间步
    impact_delays: Dict[str, Dict[str, int]]
    # 影响强度范围
    strength_ranges: Dict[str, Dict[str, Tuple[float, float]]]
    
    def get_influence(self, source_domain: str, target_domain: str) -> float:
        """获取源领域对目标领域的影响强度"""
        return self.matrix.get(source_domain, {}).get(target_domain, 0.0)
    
    def get_impact_type(self, source_domain: str, target_domain: str) -> str:
        """获取源领域对目标领域的影响类型"""
        return self.impact_types.get(source_domain, {}).get(target_domain, "neutral")
    
    def get_impact_delay(self, source_domain: str, target_domain: str) -> int:
        """获取源领域对目标领域的影响延迟"""
        return self.impact_delays.get(source_domain, {}).get(target_domain, 0)
    
    def get_strength_range(self, source_domain: str, target_domain: str) -> Tuple[float, float]:
        """获取源领域对目标领域的影响强度范围"""
        return self.strength_ranges.get(source_domain, {}).get(target_domain, (0.0, 1.0))
    
    def update_influence(self, source_domain: str, target_domain: str, strength: float) -> None:
        """更新影响强度"""
        if source_domain not in self.matrix:
            self.matrix[source_domain] = {}
        self.matrix[source_domain][target_domain] = strength
    
    def normalize(self) -> None:
        """归一化影响矩阵，确保每行和为1"""
        for source_domain, targets in self.matrix.items():
            total = sum(abs(v) for v in targets.values())
            if total > 0:
                for target_domain in targets:
                    self.matrix[source_domain][target_domain] /= total


@dataclass
class InteractionEffect:
    """交互效应记录"""
    source_domain: str
    target_domain: str
    effect_value: float
    effect_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    applied: bool = False
    
    def apply_effect(self, target_state: Dict[str, Any]) -> Dict[str, Any]:
        """应用效应到目标状态"""
        # 创建状态副本以避免修改原状态
        new_state = target_state.copy()
        
        # 根据效应类型应用不同的处理逻辑
        if self.effect_type == "positive":
            # 正面影响通常增强相关指标
            for key, value in new_state.items():
                if isinstance(value, (int, float)):
                    new_state[key] = value * (1 + abs(self.effect_value))
        elif self.effect_type == "negative":
            # 负面影响通常减弱相关指标
            for key, value in new_state.items():
                if isinstance(value, (int, float)):
                    new_state[key] = value * (1 - abs(self.effect_value))
        elif self.effect_type == "neutral":
            # 中性影响可能只产生波动
            for key, value in new_state.items():
                if isinstance(value, (int, float)):
                    new_state[key] = value * (1 + self.effect_value)
        
        self.applied = True
        return new_state


@dataclass
class NonlinearFeedback:
    """非线性反馈机制"""
    feedback_type: str  # 'exponential', 'logistic', 'threshold', 'oscillatory'
    parameters: Dict[str, float]  # 反馈参数
    
    def calculate_feedback(self, value: float) -> float:
        """根据反馈类型计算反馈值"""
        if self.feedback_type == "exponential":
            # 指数反馈：f(x) = a * e^(b*x)
            a = self.parameters.get("a", 1.0)
            b = self.parameters.get("b", 0.1)
            return a * np.exp(b * value)
        
        elif self.feedback_type == "logistic":
            # 逻辑斯蒂反馈：f(x) = L / (1 + e^(-k*(x-x0)))
            L = self.parameters.get("L", 1.0)  # 最大值
            k = self.parameters.get("k", 1.0)  # 增长速率
            x0 = self.parameters.get("x0", 0.0)  # 中点
            return L / (1 + np.exp(-k * (value - x0)))
        
        elif self.feedback_type == "threshold":
            # 阈值反馈：当超过阈值时产生突变
            threshold = self.parameters.get("threshold", 0.5)
            max_effect = self.parameters.get("max_effect", 1.0)
            if value >= threshold:
                return max_effect
            return 0.0
        
        elif self.feedback_type == "oscillatory":
            # 振荡反馈：产生周期性变化
            amplitude = self.parameters.get("amplitude", 1.0)
            frequency = self.parameters.get("frequency", 1.0)
            phase = self.parameters.get("phase", 0.0)
            return amplitude * np.sin(frequency * value + phase)
        
        # 默认线性反馈
        return value


@dataclass
class CascadeEvent:
    """级联效应事件"""
    source_domain: str
    source_change: float
    affected_domains: List[str]
    severity: float  # 0.0-1.0
    propagation_path: List[Tuple[str, str]] = field(default_factory=list)
    effects: Dict[str, float] = field(default_factory=dict)
    
    def calculate_cascade_effects(self, impact_matrix: ImpactMatrix) -> Dict[str, float]:
        """计算级联效应在各领域的影响"""
        effects = {}
        visited = set()
        
        def propagate(source: str, current_impact: float, path: List[Tuple[str, str]]):
            """递归传播影响"""
            if source in visited or current_impact < 0.01:  # 影响太小则停止传播
                return
            
            visited.add(source)
            
            # 获取所有可能的目标领域
            for target, strength in impact_matrix.matrix.get(source, {}).items():
                if target in visited:
                    continue
                
                # 计算在目标领域的影响
                impact = current_impact * strength * self.severity
                effects[target] = effects.get(target, 0.0) + impact
                
                # 记录传播路径
                new_path = path + [(source, target)]
                
                # 递归传播
                propagate(target, impact, new_path)
        
        # 开始传播
        propagate(self.source_domain, self.source_change, [])
        
        self.effects = effects
        return effects


class InteractionModel:
    """领域交互模型"""
    
    def __init__(self,
                 domains: List[str],
                 impact_matrix: Optional[ImpactMatrix] = None,
                 random_seed: Optional[int] = None,
                 simulation_mode: SimulationMode = SimulationMode.REALISTIC):
        """初始化交互模型"""
        self.logger = get_logger(__name__)
        self.domains = domains
        self.random = RandomGenerator(seed=random_seed)
        self.simulation_mode = simulation_mode
        
        # 初始化影响矩阵
        self.impact_matrix = impact_matrix or self._create_default_impact_matrix()
        
        # 初始化非线性反馈机制
        self.feedback_mechanisms = self._create_default_feedback_mechanisms()
        
        # 初始化交互历史
        self.interaction_history: List[InteractionEffect] = []
        
        # 初始化待应用的效应队列（考虑延迟）
        self.pending_effects: List[Tuple[int, InteractionEffect]] = []  # (delay_steps, effect)
        
        # 初始化级联事件跟踪
        self.cascade_events: List[CascadeEvent] = []
        
        # 初始化领域状态缓存
        self.domain_states: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"交互模型初始化完成，包含 {len(domains)} 个领域")
    
    def _create_default_impact_matrix(self) -> ImpactMatrix:
        """创建默认的影响矩阵"""
        # 定义领域间的基本影响关系
        matrix = {}
        impact_types = {}
        impact_delays = {}
        strength_ranges = {}
        
        # 为每个领域初始化
        for domain in self.domains:
            matrix[domain] = {}
            impact_types[domain] = {}
            impact_delays[domain] = {}
            strength_ranges[domain] = {}
        
        # 经济 -> 其他领域的影响
        if "economy" in self.domains:
            matrix["economy"]["politics"] = 0.3  # 经济影响政治
            matrix["economy"]["technology"] = 0.4  # 经济影响技术
            matrix["economy"]["climate"] = 0.2  # 经济影响气候
            
            impact_types["economy"]["politics"] = "positive"
            impact_types["economy"]["technology"] = "positive"
            impact_types["economy"]["climate"] = "negative"
            
            impact_delays["economy"]["politics"] = 1
            impact_delays["economy"]["technology"] = 2
            impact_delays["economy"]["climate"] = 3
            
            strength_ranges["economy"]["politics"] = (0.1, 0.5)
            strength_ranges["economy"]["technology"] = (0.2, 0.6)
            strength_ranges["economy"]["climate"] = (0.1, 0.3)
        
        # 政治 -> 其他领域的影响
        if "politics" in self.domains:
            matrix["politics"]["economy"] = 0.4  # 政治影响经济
            matrix["politics"]["technology"] = 0.3  # 政治影响技术
            matrix["politics"]["climate"] = 0.5  # 政治影响气候
            
            impact_types["politics"]["economy"] = "positive"
            impact_types["politics"]["technology"] = "positive"
            impact_types["politics"]["climate"] = "positive"
            
            impact_delays["politics"]["economy"] = 1
            impact_delays["politics"]["technology"] = 2
            impact_delays["politics"]["climate"] = 1
            
            strength_ranges["politics"]["economy"] = (0.2, 0.6)
            strength_ranges["politics"]["technology"] = (0.1, 0.5)
            strength_ranges["politics"]["climate"] = (0.3, 0.7)
        
        # 技术 -> 其他领域的影响
        if "technology" in self.domains:
            matrix["technology"]["economy"] = 0.5  # 技术影响经济
            matrix["technology"]["politics"] = 0.2  # 技术影响政治
            matrix["technology"]["climate"] = 0.4  # 技术影响气候
            
            impact_types["technology"]["economy"] = "positive"
            impact_types["technology"]["politics"] = "neutral"
            impact_types["technology"]["climate"] = "positive"  # 技术发展可能有助于解决气候问题
            
            impact_delays["technology"]["economy"] = 3
            impact_delays["technology"]["politics"] = 2
            impact_delays["technology"]["climate"] = 4
            
            strength_ranges["technology"]["economy"] = (0.3, 0.7)
            strength_ranges["technology"]["politics"] = (0.0, 0.4)
            strength_ranges["technology"]["climate"] = (0.2, 0.6)
        
        # 气候 -> 其他领域的影响
        if "climate" in self.domains:
            matrix["climate"]["economy"] = 0.3  # 气候影响经济
            matrix["climate"]["politics"] = 0.2  # 气候影响政治
            matrix["climate"]["technology"] = 0.3  # 气候影响技术（推动创新）
            
            impact_types["climate"]["economy"] = "negative"  # 气候变化通常对经济有负面影响
            impact_types["climate"]["politics"] = "negative"  # 气候变化可能加剧政治紧张
            impact_types["climate"]["technology"] = "positive"  # 气候变化可能推动绿色技术创新
            
            impact_delays["climate"]["economy"] = 2
            impact_delays["climate"]["politics"] = 3
            impact_delays["climate"]["technology"] = 4
            
            strength_ranges["climate"]["economy"] = (0.1, 0.5)
            strength_ranges["climate"]["politics"] = (0.1, 0.4)
            strength_ranges["climate"]["technology"] = (0.2, 0.5)
        
        # 归一化矩阵
        impact_matrix = ImpactMatrix(matrix, impact_types, impact_delays, strength_ranges)
        impact_matrix.normalize()
        
        return impact_matrix
    
    def _create_default_feedback_mechanisms(self) -> Dict[str, NonlinearFeedback]:
        """创建默认的非线性反馈机制"""
        feedbacks = {
            # 经济增长的饱和效应（逻辑斯蒂增长）
            "economy_growth": NonlinearFeedback(
                feedback_type="logistic",
                parameters={"L": 1.0, "k": 1.0, "x0": 0.5}
            ),
            # 政治稳定性的阈值效应
            "political_stability": NonlinearFeedback(
                feedback_type="threshold",
                parameters={"threshold": 0.3, "max_effect": 1.0}
            ),
            # 技术创新的指数加速
            "tech_innovation": NonlinearFeedback(
                feedback_type="exponential",
                parameters={"a": 1.0, "b": 0.2}
            ),
            # 气候变化的振荡效应（El Nino等周期）
            "climate_change": NonlinearFeedback(
                feedback_type="oscillatory",
                parameters={"amplitude": 0.2, "frequency": 0.5, "phase": 0.0}
            )
        }
        return feedbacks
    
    def update_domain_state(self, domain: str, state: Dict[str, Any]) -> None:
        """更新领域状态"""
        self.domain_states[domain] = state.copy()
        self.logger.debug(f"更新领域状态: {domain}")
    
    def calculate_interactions(self, time_step: int) -> List[InteractionEffect]:
        """计算所有领域间的交互效应"""
        effects = []
        
        # 遍历所有源领域
        for source_domain in self.domains:
            if source_domain not in self.domain_states:
                continue
            
            # 获取源领域状态
            source_state = self.domain_states[source_domain]
            
            # 遍历所有可能的目标领域
            for target_domain in self.domains:
                if source_domain == target_domain or target_domain not in self.domain_states:
                    continue
                
                # 获取影响参数
                influence_strength = self.impact_matrix.get_influence(source_domain, target_domain)
                if influence_strength <= 0:
                    continue
                
                impact_type = self.impact_matrix.get_impact_type(source_domain, target_domain)
                impact_delay = self.impact_matrix.get_impact_delay(source_domain, target_domain)
                
                # 计算基础效应值
                base_effect = self._calculate_base_effect(source_domain, source_state, target_domain)
                
                # 应用非线性反馈
                feedback_effect = self._apply_feedback(source_domain, base_effect)
                
                # 最终效应值
                effect_value = influence_strength * feedback_effect
                
                # 根据模拟模式调整随机性
                if self.simulation_mode == SimulationMode.REALISTIC:
                    # 添加一些随机波动
                    random_factor = self.random.normal(1.0, 0.1)
                    effect_value *= random_factor
                elif self.simulation_mode == SimulationMode.OPTIMISTIC:
                    # 乐观模式：增强正面影响，减弱负面影响
                    if impact_type == "positive":
                        effect_value *= 1.2
                    elif impact_type == "negative":
                        effect_value *= 0.8
                elif self.simulation_mode == SimulationMode.PESSIMISTIC:
                    # 悲观模式：减弱正面影响，增强负面影响
                    if impact_type == "positive":
                        effect_value *= 0.8
                    elif impact_type == "negative":
                        effect_value *= 1.2
                
                # 限制效应值范围
                min_strength, max_strength = self.impact_matrix.get_strength_range(source_domain, target_domain)
                effect_value = max(min_strength, min(max_strength, effect_value))
                
                # 创建交互效应
                effect = InteractionEffect(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    effect_value=effect_value,
                    effect_type=impact_type
                )
                
                if impact_delay == 0:
                    # 立即应用的效应
                    effects.append(effect)
                else:
                    # 延迟应用的效应
                    self.pending_effects.append((impact_delay, effect))
        
        # 处理待应用的效应
        current_step_effects = self._process_pending_effects()
        effects.extend(current_step_effects)
        
        # 记录交互历史
        self.interaction_history.extend(effects)
        
        return effects
    
    def _calculate_base_effect(self, source_domain: str, source_state: Dict[str, Any], target_domain: str) -> float:
        """计算基础效应值"""
        # 根据源领域和目标领域的关系计算基础效应
        # 这里使用简化的方法，实际应用中可以更加复杂
        
        # 获取源领域的关键指标变化
        key_metrics = self._get_key_metrics(source_domain)
        
        # 计算指标加权平均值作为效应基础
        weights = {}
        if source_domain == "economy":
            weights = {"gdp_growth": 0.4, "inflation": 0.3, "unemployment": 0.3}
        elif source_domain == "politics":
            weights = {"stability": 0.6, "polarization": 0.4}
        elif source_domain == "technology":
            weights = {"innovation_index": 0.5, "adoption_rate": 0.5}
        elif source_domain == "climate":
            weights = {"temperature_change": 0.4, "extreme_events": 0.6}
        
        # 计算加权平均变化
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in source_state and isinstance(source_state[metric], (int, float)):
                weighted_sum += source_state[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        
        # 默认值
        return 0.1
    
    def _get_key_metrics(self, domain: str) -> List[str]:
        """获取领域的关键指标"""
        metrics = {
            "economy": ["gdp_growth", "inflation", "unemployment", "productivity", "investment"],
            "politics": ["stability", "polarization", "corruption", "democracy_index", "policy_effectiveness"],
            "technology": ["innovation_index", "adoption_rate", "r_d_investment", "patents", "digital_infrastructure"],
            "climate": ["temperature_change", "extreme_events", "co2_level", "sea_level_rise", "biodiversity_loss"]
        }
        return metrics.get(domain, [])
    
    def _apply_feedback(self, source_domain: str, base_effect: float) -> float:
        """应用非线性反馈"""
        # 根据源领域选择适当的反馈机制
        if source_domain == "economy" and "economy_growth" in self.feedback_mechanisms:
            return self.feedback_mechanisms["economy_growth"].calculate_feedback(base_effect)
        elif source_domain == "politics" and "political_stability" in self.feedback_mechanisms:
            return self.feedback_mechanisms["political_stability"].calculate_feedback(base_effect)
        elif source_domain == "technology" and "tech_innovation" in self.feedback_mechanisms:
            return self.feedback_mechanisms["tech_innovation"].calculate_feedback(base_effect)
        elif source_domain == "climate" and "climate_change" in self.feedback_mechanisms:
            return self.feedback_mechanisms["climate_change"].calculate_feedback(base_effect)
        
        # 默认返回基础效应
        return base_effect
    
    def _process_pending_effects(self) -> List[InteractionEffect]:
        """处理待应用的效应"""
        current_effects = []
        remaining_effects = []
        
        # 检查每个待应用的效应
        for delay, effect in self.pending_effects:
            if delay <= 1:
                # 延迟已到，应该应用
                current_effects.append(effect)
            else:
                # 减少延迟计数，保留到下一轮
                remaining_effects.append((delay - 1, effect))
        
        # 更新待处理队列
        self.pending_effects = remaining_effects
        
        return current_effects
    
    def apply_effects(self, effects: List[InteractionEffect]) -> Dict[str, Dict[str, Any]]:
        """应用所有交互效应到领域状态"""
        updated_states = {}
        
        # 按目标领域分组效应
        effects_by_target: Dict[str, List[InteractionEffect]] = {}
        for effect in effects:
            if effect.target_domain not in effects_by_target:
                effects_by_target[effect.target_domain] = []
            effects_by_target[effect.target_domain].append(effect)
        
        # 应用每个目标领域的效应
        for target_domain, target_effects in effects_by_target.items():
            if target_domain not in self.domain_states:
                continue
            
            # 获取目标领域当前状态
            current_state = self.domain_states[target_domain].copy()
            
            # 依次应用每个效应
            for effect in target_effects:
                current_state = effect.apply_effect(current_state)
            
            # 检查是否需要触发级联效应
            if self._should_trigger_cascade(target_domain, current_state):
                self._trigger_cascade_effect(target_domain, current_state)
            
            # 更新状态
            updated_states[target_domain] = current_state
            self.update_domain_state(target_domain, current_state)
        
        return updated_states
    
    def _should_trigger_cascade(self, domain: str, state: Dict[str, Any]) -> bool:
        """判断是否应该触发级联效应"""
        # 检查关键指标是否超过阈值
        thresholds = {
            "economy": {"gdp_growth": -0.05, "inflation": 0.1, "unemployment": 0.2},
            "politics": {"stability": 0.3, "polarization": 0.8},
            "climate": {"temperature_change": 2.0, "extreme_events": 5.0}
        }
        
        domain_thresholds = thresholds.get(domain, {})
        for metric, threshold in domain_thresholds.items():
            if metric in state and isinstance(state[metric], (int, float)):
                if state[metric] >= threshold:  # 对于负面指标，超过阈值触发级联
                    self.logger.warning(f"领域 {domain} 指标 {metric} 超过阈值 {threshold}，可能触发级联效应")
                    return True
                elif state[metric] <= -threshold:  # 对于增长率等，大幅下降也触发级联
                    self.logger.warning(f"领域 {domain} 指标 {metric} 大幅下降至 {state[metric]}，可能触发级联效应")
                    return True
        
        return False
    
    def _trigger_cascade_effect(self, source_domain: str, state: Dict[str, Any]) -> None:
        """触发级联效应"""
        # 计算源领域变化幅度
        source_change = 0.0
        key_metrics = self._get_key_metrics(source_domain)
        
        for metric in key_metrics[:3]:  # 只使用前3个关键指标
            if metric in state and isinstance(state[metric], (int, float)):
                source_change += abs(state[metric])
        
        # 确定受影响的领域
        affected_domains = []
        for domain in self.domains:
            if domain != source_domain:
                strength = self.impact_matrix.get_influence(source_domain, domain)
                if strength > 0.1:  # 只有足够强的影响才会被计入
                    affected_domains.append(domain)
        
        # 计算级联严重程度
        severity = min(1.0, source_change * 0.2)  # 限制在0-1之间
        
        # 创建级联事件
        cascade_event = CascadeEvent(
            source_domain=source_domain,
            source_change=source_change,
            affected_domains=affected_domains,
            severity=severity
        )
        
        # 计算级联效应
        cascade_event.calculate_cascade_effects(self.impact_matrix)
        
        # 记录级联事件
        self.cascade_events.append(cascade_event)
        self.logger.warning(f"触发级联效应：源领域 {source_domain}，严重程度 {severity:.2f}，影响领域数 {len(affected_domains)}")
    
    def step(self, time_step: int) -> Dict[str, Dict[str, Any]]:
        """执行一步交互计算"""
        # 计算交互效应
        effects = self.calculate_interactions(time_step)
        
        # 应用效应
        updated_states = self.apply_effects(effects)
        
        # 记录统计信息
        self.logger.debug(f"时间步 {time_step}：计算了 {len(effects)} 个交互效应，更新了 {len(updated_states)} 个领域状态")
        
        return updated_states
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """获取交互摘要"""
        summary = {
            "total_interactions": len(self.interaction_history),
            "pending_effects": len(self.pending_effects),
            "cascade_events": len(self.cascade_events),
            "domains": self.domains,
            "interaction_matrix": self.impact_matrix.matrix
        }
        
        # 按源领域和目标领域统计交互次数
        interaction_counts = {}
        for effect in self.interaction_history:
            key = f"{effect.source_domain}->{effect.target_domain}"
            interaction_counts[key] = interaction_counts.get(key, 0) + 1
        summary["interaction_counts"] = interaction_counts
        
        # 统计级联事件详情
        if self.cascade_events:
            cascade_details = []
            for event in self.cascade_events:
                cascade_details.append({
                    "source_domain": event.source_domain,
                    "severity": event.severity,
                    "affected_domains": list(event.effects.keys()),
                    "max_effect": max(event.effects.values()) if event.effects else 0
                })
            summary["cascade_details"] = cascade_details
        
        return summary
    
    def visualize_impact_matrix(self, output_path: Optional[str] = None) -> None:
        """可视化影响矩阵"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 准备数据
            matrix_data = []
            for source in self.domains:
                row = []
                for target in self.domains:
                    row.append(self.impact_matrix.get_influence(source, target))
                matrix_data.append(row)
            
            # 创建热力图
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                matrix_data, 
                xticklabels=self.domains, 
                yticklabels=self.domains,
                annot=True, 
                cmap="coolwarm", 
                vmin=-1, 
                vmax=1,
                fmt=".2f"
            )
            plt.title("领域间影响矩阵")
            plt.xlabel("目标领域")
            plt.ylabel("源领域")
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                self.logger.info(f"影响矩阵可视化已保存到 {output_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("无法导入可视化库，请安装 matplotlib 和 seaborn")
        except Exception as e:
            self.logger.error(f"可视化影响矩阵时出错: {e}")
    
    def calculate_system_stability(self) -> float:
        """计算系统整体稳定性"""
        # 使用影响矩阵的特征值分析来评估系统稳定性
        try:
            # 将字典转换为numpy矩阵
            n = len(self.domains)
            matrix_array = np.zeros((n, n))
            
            for i, source in enumerate(self.domains):
                for j, target in enumerate(self.domains):
                    matrix_array[i, j] = self.impact_matrix.get_influence(source, target)
            
            # 计算特征值
            eigenvalues = np.linalg.eigvals(matrix_array)
            
            # 系统稳定性指标：所有特征值的模的最大值小于1
            max_eigenvalue = max(abs(ev) for ev in eigenvalues)
            
            # 稳定性得分（0-1），特征值越小越稳定
            stability = max(0.0, 1.0 - max_eigenvalue)
            
            return stability
            
        except Exception as e:
            self.logger.error(f"计算系统稳定性时出错: {e}")
            return 0.5  # 返回中性稳定性作为默认值
    
    def reset(self) -> None:
        """重置交互模型"""
        self.interaction_history = []
        self.pending_effects = []
        self.cascade_events = []
        self.domain_states = {}
        self.logger.info("交互模型已重置")


# 工厂函数，用于创建预配置的交互模型
def create_standard_interaction_model(random_seed: Optional[int] = None) -> InteractionModel:
    """创建标准配置的交互模型"""
    domains = ["economy", "politics", "technology", "climate"]
    model = InteractionModel(domains=domains, random_seed=random_seed)
    return model


def create_economic_focus_model(random_seed: Optional[int] = None) -> InteractionModel:
    """创建经济重点的交互模型"""
    domains = ["economy", "politics", "technology", "climate"]
    
    # 创建默认矩阵
    model = InteractionModel(domains=domains, random_seed=random_seed)
    
    # 增强经济领域的影响力
    model.impact_matrix.update_influence("economy", "politics", 0.5)
    model.impact_matrix.update_influence("economy", "technology", 0.6)
    model.impact_matrix.update_influence("economy", "climate", 0.3)
    
    return model


def create_climate_focus_model(random_seed: Optional[int] = None) -> InteractionModel:
    """创建气候重点的交互模型"""
    domains = ["economy", "politics", "technology", "climate"]
    
    # 创建默认矩阵
    model = InteractionModel(domains=domains, random_seed=random_seed)
    
    # 增强气候领域的影响力
    model.impact_matrix.update_influence("climate", "economy", 0.5)
    model.impact_matrix.update_influence("climate", "politics", 0.4)
    model.impact_matrix.update_influence("climate", "technology", 0.5)
    
    # 增强其他领域对气候的影响
    model.impact_matrix.update_influence("economy", "climate", 0.3)
    model.impact_matrix.update_influence("technology", "climate", 0.6)
    
    return model