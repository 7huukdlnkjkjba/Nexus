#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phantom Crawler Nexus - 世界线模拟器
事件系统模块（优化版）
负责生成和管理影响各个领域的重大事件
"""

import logging
import random
import hashlib
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
import math
from functools import lru_cache

from ..utils.logger import get_logger
from ..utils.random_utils import get_seeded_random
from ..data.data_types import Event, EventType, EventSeverity, Region


class EventSystem:
    """
    全球事件系统
    生成和管理影响各个领域的重大事件
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化事件系统（优化版）
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("EventSystem")
        self.config = config
        
        # 事件配置
        self.event_probabilities = config.get("event_probabilities", {
            "economic_crisis": 0.05,        # 经济危机概率
            "technological_breakthrough": 0.08,  # 技术突破概率
            "conflict": 0.06,              # 冲突概率
            "pandemic": 0.02,              # 大流行病概率
            "natural_disaster": 0.12,      # 自然灾害概率
            "political_crisis": 0.07,      # 政治危机概率
            "resource_crisis": 0.04,       # 资源危机概率
            "social_unrest": 0.09          # 社会动荡概率
        })
        
        # 地区定义
        self.regions = config.get("regions", [
            "North_America",
            "South_America",
            "Europe",
            "Africa",
            "Asia",
            "Oceania",
            "Middle_East"
        ])
        
        # 国家定义（主要大国）
        self.major_countries = config.get("major_countries", [
            "United_States",
            "China",
            "Russia",
            "India",
            "Japan",
            "Germany",
            "United_Kingdom",
            "France",
            "Brazil",
            "Canada",
            "Australia",
            "South_Korea"
        ])
        
        # 事件历史记录（限制大小）
        self.max_history_size = config.get("max_history_size", 1000)
        self.event_history = []
        
        # 事件缓存（用于避免重复生成相似条件的事件）
        self.event_cache = {}
        self.max_cache_size = config.get("max_cache_size", 1000)
        
        # 事件生成器注册
        self.event_generators = {
            "economic_crisis": self._generate_economic_crisis,
            "technological_breakthrough": self._generate_technological_breakthrough,
            "conflict": self._generate_conflict,
            "pandemic": self._generate_pandemic,
            "natural_disaster": self._generate_natural_disaster,
            "political_crisis": self._generate_political_crisis,
            "resource_crisis": self._generate_resource_crisis,
            "social_unrest": self._generate_social_unrest
        }
        
        # 事件影响计算器注册
        self.impact_calculators = {
            "economic_crisis": self._calculate_economic_crisis_impact,
            "technological_breakthrough": self._calculate_technological_breakthrough_impact,
            "conflict": self._calculate_conflict_impact,
            "pandemic": self._calculate_pandemic_impact,
            "natural_disaster": self._calculate_natural_disaster_impact,
            "political_crisis": self._calculate_political_crisis_impact,
            "resource_crisis": self._calculate_resource_crisis_impact,
            "social_unrest": self._calculate_social_unrest_impact
        }
        
        self.logger.info("事件系统初始化完成")
    
    def _generate_context_hash(self, current_time: datetime, 
                              economic_state: Optional[Dict[str, Any]],
                              political_state: Optional[Dict[str, Any]],
                              technological_state: Optional[Dict[str, Any]],
                              climate_state: Optional[Dict[str, Any]],
                              seed: Optional[int]) -> str:
        """
        生成上下文的哈希值，用于缓存查找
        
        Args:
            current_time: 当前时间
            economic_state: 经济状态
            political_state: 政治状态
            technological_state: 技术状态
            climate_state: 气候状态
            seed: 随机种子
            
        Returns:
            上下文哈希值
        """
        # 提取关键上下文信息用于哈希计算
        key_info = {
            "year": current_time.year,
            "month": current_time.month,
            "economic_metrics": {
                "gdp_growth": economic_state.get("global_gdp_growth", 0) if economic_state else 0,
                "inflation": economic_state.get("global_inflation", 0) if economic_state else 0,
                "unemployment": economic_state.get("global_unemployment", 0) if economic_state else 0
            } if economic_state else {},
            "political_stability": political_state.get("global_stability", 0) if political_state else 0,
            "tech_progress": technological_state.get("progress_index", 0) if technological_state else 0,
            "climate_risk": climate_state.get("risk_level", 0) if climate_state else 0,
            "seed": seed
        }
        
        # 将关键信息转换为字符串并生成哈希
        key_str = str(key_info)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def generate_events(self, current_time: datetime, 
                      economic_state: Optional[Dict[str, Any]] = None,
                      political_state: Optional[Dict[str, Any]] = None,
                      technological_state: Optional[Dict[str, Any]] = None,
                      climate_state: Optional[Dict[str, Any]] = None,
                      seed: Optional[int] = None) -> List[Event]:
        """
        生成新事件（优化版，使用缓存机制）
        
        Args:
            current_time: 当前时间
            economic_state: 经济状态
            political_state: 政治状态
            technological_state: 技术状态
            climate_state: 气候状态
            seed: 随机种子
            
        Returns:
            生成的事件列表
        """
        # 生成上下文哈希，用于缓存查找
        context_hash = self._generate_context_hash(current_time, economic_state, 
                                                 political_state, technological_state, 
                                                 climate_state, seed)
        
        # 检查缓存
        if context_hash in self.event_cache:
            self.logger.debug(f"从缓存中获取事件: {context_hash}")
            cached_events = self.event_cache[context_hash]
            # 复制缓存的事件以避免修改原始对象
            events = [Event(**event.__dict__.copy()) for event in cached_events]
            
            # 添加到历史记录
            for event in events:
                self.event_history.append(event)
            
            return events
        
        try:
            # 初始化随机数生成器
            if seed is not None:
                rng = random.Random(seed)
            else:
                rng = get_seeded_random(hash(str(current_time)))
            
            # 批量收集需要生成的事件类型
            event_types_to_generate = []
            for event_type, base_prob in self.event_probabilities.items():
                # 调整概率基于当前状态
                adjusted_prob = self._adjust_event_probability(
                    event_type, base_prob, economic_state, political_state,
                    technological_state, climate_state, current_time
                )
                
                if rng.random() < adjusted_prob:
                    event_types_to_generate.append(event_type)
            
            # 批量生成事件
            events = []
            for event_type in event_types_to_generate:
                if event_type in self.event_generators:
                    event = self.event_generators[event_type](
                        current_time, economic_state, political_state,
                        technological_state, climate_state, rng
                    )
                    
                    if event:
                        # 计算事件影响
                        event = self._calculate_event_impacts(
                            event, economic_state, political_state,
                            technological_state, climate_state
                        )
                        events.append(event)
            
            # 确保一年内生成的事件数量合理
            max_events_per_year = self.config.get("max_events_per_year", 5)
            if len(events) > max_events_per_year:
                # 按严重程度排序，保留最重要的事件
                events.sort(key=lambda e: e.severity.value, reverse=True)
                events = events[:max_events_per_year]
            
            # 添加到历史记录（并限制历史记录大小）
            for event in events:
                self.event_history.append(event)
            
            # 清理旧的历史记录
            if len(self.event_history) > self.max_history_size:
                self.event_history = self.event_history[-self.max_history_size:]
            
            # 缓存结果
            self.event_cache[context_hash] = events.copy()
            
            # 清理缓存
            if len(self.event_cache) > self.max_cache_size:
                # 删除最旧的缓存项
                oldest_keys = list(self.event_cache.keys())[:-self.max_cache_size]
                for key in oldest_keys:
                    del self.event_cache[key]
            
            self.logger.info(f"生成了 {len(events)} 个新事件")
            return events
            
        except Exception as e:
            self.logger.error(f"生成事件失败: {str(e)}")
            return []
    
    def batch_generate_events(self, event_requests: List[Dict[str, Any]]) -> List[List[Event]]:
        """
        批量生成事件，优化多条世界线的事件生成性能
        
        Args:
            event_requests: 事件请求列表，每个请求包含生成事件所需的参数
            
        Returns:
            每个请求对应的事件列表
        """
        results = []
        
        # 为每个请求生成事件
        for request in event_requests:
            # 提取请求参数
            current_time = request.get("current_time")
            economic_state = request.get("economic_state")
            political_state = request.get("political_state")
            technological_state = request.get("technological_state")
            climate_state = request.get("climate_state")
            seed = request.get("seed")
            
            # 生成事件
            events = self.generate_events(
                current_time, economic_state, political_state,
                technological_state, climate_state, seed
            )
            results.append(events)
        
        return results
    
    def _adjust_event_probability(self, event_type: str, base_prob: float,
                                economic_state: Optional[Dict[str, Any]],
                                political_state: Optional[Dict[str, Any]],
                                technological_state: Optional[Dict[str, Any]],
                                climate_state: Optional[Dict[str, Any]],
                                current_time: datetime) -> float:
        """
        调整事件概率
        
        Args:
            event_type: 事件类型
            base_prob: 基础概率
            economic_state: 经济状态
            political_state: 政治状态
            technological_state: 技术状态
            climate_state: 气候状态
            current_time: 当前时间
            
        Returns:
            调整后的概率
        """
        adjusted_prob = base_prob
        
        # 经济危机概率调整
        if event_type == "economic_crisis" and economic_state:
            # 经济衰退增加经济危机概率
            if economic_state.get("global_gdp_growth") is not None:
                gdp_growth = economic_state["global_gdp_growth"]
                if gdp_growth < 0:
                    # 经济衰退时概率增加
                    adjusted_prob *= (1 + abs(gdp_growth) * 2)
            
            # 债务水平高增加危机概率
            debt_level = economic_state.get("global_debt_level", 200)
            if debt_level > 300:
                adjusted_prob *= 1.5
        
        # 技术突破概率调整
        elif event_type == "technological_breakthrough" and technological_state:
            # 研发投入增加突破概率
            rd_investment = technological_state.get("global_rd_investment", 2.5)
            # rd_investment 是 GDP 的百分比
            if rd_investment > 3.5:
                adjusted_prob *= 1.2
            
            # 技术成熟度接近临界点时增加突破概率
            for tech, data in technological_state.get("technologies", {}).items():
                maturity = data.get("maturity", 0)
                if 0.7 <= maturity < 0.9:
                    adjusted_prob *= 1.1
        
        # 冲突概率调整
        elif event_type == "conflict" and political_state:
            # 政治紧张局势增加冲突概率
            global_stability = political_state.get("global_stability", 0.7)
            if global_stability < 0.5:
                adjusted_prob *= 1.5
        
        # 自然灾害概率调整
        elif event_type == "natural_disaster" and climate_state:
            # 气候变化增加自然灾害概率
            temp_increase = climate_state.get("temperature", {}).get("global_mean", 1.2)
            if temp_increase > 1.5:
                adjusted_prob *= 1.3
        
        # 政治危机概率调整
        elif event_type == "political_crisis" and political_state:
            # 政治不稳定性增加危机概率
            for country, data in political_state.get("countries", {}).items():
                stability = data.get("stability", 0.7)
                if stability < 0.4:
                    adjusted_prob *= 1.2
        
        # 资源危机概率调整
        elif event_type == "resource_crisis":
            # 能源价格波动增加资源危机概率
            if economic_state:
                energy_prices = economic_state.get("energy_prices", {})
                oil_price = energy_prices.get("oil", 80)
                if oil_price > 150 or oil_price < 30:
                    adjusted_prob *= 1.3
        
        # 社会动荡概率调整
        elif event_type == "social_unrest":
            # 经济不平等增加社会动荡概率
            if economic_state:
                inequality = economic_state.get("global_inequality", 0.4)
                if inequality > 0.6:
                    adjusted_prob *= 1.4
        
        # 时间因素：某些事件有周期性
        year = current_time.year
        
        # 选举年份可能增加政治事件概率
        if event_type in ["political_crisis", "social_unrest"] and (year % 4 == 0 or year % 5 == 0):
            adjusted_prob *= 1.1
        
        # 确保概率在合理范围内
        adjusted_prob = min(adjusted_prob, 0.5)  # 最大50%概率
        adjusted_prob = max(adjusted_prob, 0.001)  # 最小0.1%概率
        
        return adjusted_prob
    
    def _generate_economic_crisis(self, current_time: datetime,
                                economic_state: Optional[Dict[str, Any]],
                                political_state: Optional[Dict[str, Any]],
                                technological_state: Optional[Dict[str, Any]],
                                climate_state: Optional[Dict[str, Any]],
                                rng: random.Random) -> Optional[Event]:
        """
        生成经济危机事件
        """
        crisis_types = [
            "financial_crisis",
            "recession",
            "inflation_crisis",
            "debt_crisis",
            "currency_crisis",
            "housing_bubble"
        ]
        
        crisis_type = rng.choice(crisis_types)
        region = rng.choice(self.regions)
        
        # 严重程度
        severity_roll = rng.random()
        if severity_roll < 0.1:
            severity = EventSeverity.EXTREME
        elif severity_roll < 0.3:
            severity = EventSeverity.MAJOR
        elif severity_roll < 0.6:
            severity = EventSeverity.MODERATE
        else:
            severity = EventSeverity.MINOR
        
        # 持续时间（年）
        duration = {
            EventSeverity.MINOR: 1,
            EventSeverity.MODERATE: 2,
            EventSeverity.MAJOR: 3,
            EventSeverity.EXTREME: 5
        }[severity] + rng.randint(0, 2)
        
        # 影响范围
        if rng.random() < 0.3:
            scope = "global"
        else:
            scope = "regional"
        
        # 描述
        descriptions = {
            "financial_crisis": f"{region}发生金融危机",
            "recession": f"{region}陷入经济衰退",
            "inflation_crisis": f"{region}面临严重通货膨胀",
            "debt_crisis": f"{region}爆发债务危机",
            "currency_crisis": f"{region}货币大幅贬值",
            "housing_bubble": f"{region}房地产泡沫破裂"
        }
        
        description = descriptions.get(crisis_type, "经济危机")
        
        event = Event(
            id=f"eco_crisis_{current_time.year}_{rng.randint(1000, 9999)}",
            type=EventType.ECONOMIC_CRISIS,
            subtype=crisis_type,
            name=f"{description} ({current_time.year})",
            description=description,
            region=region,
            severity=severity,
            start_year=current_time.year,
            duration=duration,
            scope=scope,
            metadata={
                "crisis_type": crisis_type,
                "affected_region": region
            }
        )
        
        return event
    
    def _generate_technological_breakthrough(self, current_time: datetime,
                                          economic_state: Optional[Dict[str, Any]],
                                          political_state: Optional[Dict[str, Any]],
                                          technological_state: Optional[Dict[str, Any]],
                                          climate_state: Optional[Dict[str, Any]],
                                          rng: random.Random) -> Optional[Event]:
        """
        生成技术突破事件
        """
        tech_categories = [
            "artificial_intelligence",
            "quantum_computing",
            "renewable_energy",
            "biotechnology",
            "space_technology",
            "advanced_materials",
            "nanotechnology",
            "fusion_energy",
            "autonomous_systems",
            "brain_computer_interfaces"
        ]
        
        tech_category = rng.choice(tech_categories)
        
        # 研究机构或国家
        if rng.random() < 0.7:
            # 由国家主导
            origin = rng.choice(self.major_countries)
        else:
            # 由跨国公司或研究机构主导
            institutions = ["MIT", "Stanford", "CERN", "Max_Planck", 
                          "IBM", "Google", "Microsoft", "Apple", 
                          "Tesla", "SpaceX", "OpenAI"]
            origin = rng.choice(institutions)
        
        # 影响程度
        impact_roll = rng.random()
        if impact_roll < 0.15:
            severity = EventSeverity.EXTREME  # 变革性突破
        elif impact_roll < 0.4:
            severity = EventSeverity.MAJOR     # 重大突破
        elif impact_roll < 0.7:
            severity = EventSeverity.MODERATE  # 中等突破
        else:
            severity = EventSeverity.MINOR     # 小型突破
        
        # 商业化时间（年）
        commercialization_time = {
            EventSeverity.MINOR: rng.randint(1, 3),
            EventSeverity.MODERATE: rng.randint(3, 7),
            EventSeverity.MAJOR: rng.randint(5, 10),
            EventSeverity.EXTREME: rng.randint(8, 15)
        }[severity]
        
        # 技术描述
        tech_names = {
            "artificial_intelligence": "通用人工智能",
            "quantum_computing": "实用量子计算机",
            "renewable_energy": "高效储能技术",
            "biotechnology": "基因编辑突破",
            "space_technology": "可重复使用火箭技术",
            "advanced_materials": "超导体材料",
            "nanotechnology": "纳米制造技术",
            "fusion_energy": "可控核聚变",
            "autonomous_systems": "全自动驾驶技术",
            "brain_computer_interfaces": "脑机接口技术"
        }
        
        tech_name = tech_names.get(tech_category, "技术突破")
        
        event = Event(
            id=f"tech_breakthrough_{current_time.year}_{rng.randint(1000, 9999)}",
            type=EventType.TECHNOLOGICAL_BREAKTHROUGH,
            subtype=tech_category,
            name=f"{tech_name}重大突破 ({current_time.year})",
            description=f"{origin}在{tech_name}领域取得重大技术突破",
            region="Global" if severity in [EventSeverity.MAJOR, EventSeverity.EXTREME] else origin,
            severity=severity,
            start_year=current_time.year,
            duration=1,  # 技术突破是瞬时事件，但影响持续
            scope="global" if severity in [EventSeverity.MAJOR, EventSeverity.EXTREME] else "regional",
            metadata={
                "technology": tech_category,
                "origin": origin,
                "commercialization_time": commercialization_time
            }
        )
        
        return event
    
    def _generate_conflict(self, current_time: datetime,
                          economic_state: Optional[Dict[str, Any]],
                          political_state: Optional[Dict[str, Any]],
                          technological_state: Optional[Dict[str, Any]],
                          climate_state: Optional[Dict[str, Any]],
                          rng: random.Random) -> Optional[Event]:
        """
        生成冲突事件
        """
        conflict_types = [
            "military_conflict",
            "trade_war",
            "cyber_conflict",
            "proxy_war",
            "border_dispute"
        ]
        
        conflict_type = rng.choice(conflict_types)
        
        # 选择参与国家
        if rng.random() < 0.3:
            # 大国参与
            country1, country2 = rng.sample(self.major_countries, 2)
        else:
            # 地区冲突
            regions = self.regions.copy()
            region = rng.choice(regions)
            
            # 为区域生成国家（简化）
            region_countries = {
                "Middle_East": ["Saudi_Arabia", "Iran", "Israel", "Syria", "Iraq", "Turkey"],
                "Europe": ["Ukraine", "Russia", "Poland", "France", "Germany"],
                "Asia": ["China", "India", "Pakistan", "North_Korea", "South_Korea", "Japan"],
                "Africa": ["Egypt", "South_Africa", "Nigeria", "Ethiopia"],
                "South_America": ["Brazil", "Argentina", "Venezuela", "Colombia"],
                "North_America": ["United_States", "Canada", "Mexico"],
                "Oceania": ["Australia", "New_Zealand"]
            }
            
            if region in region_countries and len(region_countries[region]) >= 2:
                country1, country2 = rng.sample(region_countries[region], 2)
            else:
                country1, country2 = rng.sample(self.major_countries, 2)
        
        # 严重程度
        severity_roll = rng.random()
        if severity_roll < 0.1:
            severity = EventSeverity.EXTREME  # 重大战争
        elif severity_roll < 0.3:
            severity = EventSeverity.MAJOR    # 严重冲突
        elif severity_roll < 0.6:
            severity = EventSeverity.MODERATE # 中等冲突
        else:
            severity = EventSeverity.MINOR    # 轻微冲突
        
        # 持续时间（年）
        duration = {
            EventSeverity.MINOR: rng.randint(1, 2),
            EventSeverity.MODERATE: rng.randint(2, 4),
            EventSeverity.MAJOR: rng.randint(3, 7),
            EventSeverity.EXTREME: rng.randint(5, 10)
        }[severity]
        
        # 冲突描述
        conflict_names = {
            "military_conflict": "军事冲突",
            "trade_war": "贸易战",
            "cyber_conflict": "网络冲突",
            "proxy_war": "代理人战争",
            "border_dispute": "边境争端"
        }
        
        conflict_name = conflict_names.get(conflict_type, "冲突")
        
        event = Event(
            id=f"conflict_{current_time.year}_{rng.randint(1000, 9999)}",
            type=EventType.CONFLICT,
            subtype=conflict_type,
            name=f"{country1}与{country2}之间的{conflict_name} ({current_time.year})",
            description=f"{country1}与{country2}之间爆发{conflict_name}",
            region=self._get_region_for_countries(country1, country2),
            severity=severity,
            start_year=current_time.year,
            duration=duration,
            scope="regional",
            metadata={
                "participants": [country1, country2],
                "conflict_type": conflict_type
            }
        )
        
        return event
    
    def _generate_pandemic(self, current_time: datetime,
                          economic_state: Optional[Dict[str, Any]],
                          political_state: Optional[Dict[str, Any]],
                          technological_state: Optional[Dict[str, Any]],
                          climate_state: Optional[Dict[str, Any]],
                          rng: random.Random) -> Optional[Event]:
        """
        生成大流行病事件
        """
        disease_types = [
            "viral_outbreak",
            "bacterial_infection",
            "zoonotic_disease",
            "antibiotic_resistant",
            "unknown_pathogen"
        ]
        
        disease_type = rng.choice(disease_types)
        
        # 起源地
        origins = self.regions.copy()
        origins.extend(["China", "India", "Brazil", "United_States", "Indonesia"])
        origin = rng.choice(origins)
        
        # 严重程度
        severity_roll = rng.random()
        if severity_roll < 0.15:
            severity = EventSeverity.EXTREME  # 全球性大流行
        elif severity_roll < 0.4:
            severity = EventSeverity.MAJOR     # 严重疫情
        elif severity_roll < 0.7:
            severity = EventSeverity.MODERATE  # 中等疫情
        else:
            severity = EventSeverity.MINOR     # 轻微疫情
        
        # 持续时间（年）
        duration = {
            EventSeverity.MINOR: rng.randint(1, 2),
            EventSeverity.MODERATE: rng.randint(2, 3),
            EventSeverity.MAJOR: rng.randint(2, 4),
            EventSeverity.EXTREME: rng.randint(3, 6)
        }[severity]
        
        # 死亡率和传播性
        mortality_rate = {
            EventSeverity.MINOR: rng.uniform(0.01, 0.1),
            EventSeverity.MODERATE: rng.uniform(0.1, 1.0),
            EventSeverity.MAJOR: rng.uniform(1.0, 5.0),
            EventSeverity.EXTREME: rng.uniform(5.0, 20.0)
        }[severity]
        
        transmission_rate = {
            EventSeverity.MINOR: rng.uniform(1.1, 2.0),
            EventSeverity.MODERATE: rng.uniform(2.0, 3.0),
            EventSeverity.MAJOR: rng.uniform(3.0, 5.0),
            EventSeverity.EXTREME: rng.uniform(5.0, 10.0)
        }[severity]
        
        # 疾病描述
        disease_names = {
            "viral_outbreak": "病毒爆发",
            "bacterial_infection": "细菌感染",
            "zoonotic_disease": "人畜共患病",
            "antibiotic_resistant": "抗生素耐药性疾病",
            "unknown_pathogen": "未知病原体"
        }
        
        disease_name = disease_names.get(disease_type, "疫情")
        
        event = Event(
            id=f"pandemic_{current_time.year}_{rng.randint(1000, 9999)}",
            type=EventType.PANDEMIC,
            subtype=disease_type,
            name=f"{origin}爆发{disease_name} ({current_time.year})",
            description=f"{origin}爆发{disease_name}并开始向全球蔓延",
            region="Global" if severity in [EventSeverity.MAJOR, EventSeverity.EXTREME] else origin,
            severity=severity,
            start_year=current_time.year,
            duration=duration,
            scope="global" if severity in [EventSeverity.MAJOR, EventSeverity.EXTREME] else "regional",
            metadata={
                "origin": origin,
                "disease_type": disease_type,
                "mortality_rate": mortality_rate,
                "transmission_rate": transmission_rate
            }
        )
        
        return event
    
    def _generate_natural_disaster(self, current_time: datetime,
                                 economic_state: Optional[Dict[str, Any]],
                                 political_state: Optional[Dict[str, Any]],
                                 technological_state: Optional[Dict[str, Any]],
                                 climate_state: Optional[Dict[str, Any]],
                                 rng: random.Random) -> Optional[Event]:
        """
        生成自然灾害事件
        """
        disaster_types = [
            "earthquake",
            "tsunami",
            "hurricane",
            "tornado",
            "flood",
            "drought",
            "wildfire",
            "volcanic_eruption",
            "heat_wave",
            "cold_wave"
        ]
        
        disaster_type = rng.choice(disaster_types)
        region = rng.choice(self.regions)
        
        # 严重程度
        severity_roll = rng.random()
        if severity_roll < 0.1:
            severity = EventSeverity.EXTREME
        elif severity_roll < 0.3:
            severity = EventSeverity.MAJOR
        elif severity_roll < 0.6:
            severity = EventSeverity.MODERATE
        else:
            severity = EventSeverity.MINOR
        
        # 持续时间（自然灾害通常持续时间较短）
        duration = {
            EventSeverity.MINOR: rng.randint(1, 3),
            EventSeverity.MODERATE: rng.randint(2, 6),
            EventSeverity.MAJOR: rng.randint(4, 12),
            EventSeverity.EXTREME: rng.randint(6, 24)
        }[severity] / 12  # 转换为年
        
        # 灾害描述
        disaster_names = {
            "earthquake": "地震",
            "tsunami": "海啸",
            "hurricane": "飓风",
            "tornado": "龙卷风",
            "flood": "洪水",
            "drought": "干旱",
            "wildfire": "野火",
            "volcanic_eruption": "火山喷发",
            "heat_wave": "热浪",
            "cold_wave": "寒潮"
        }
        
        disaster_name = disaster_names.get(disaster_type, "自然灾害")
        
        # 强度指标
        if disaster_type == "earthquake":
            # 震级
            magnitude = {
                EventSeverity.MINOR: rng.uniform(4.0, 5.0),
                EventSeverity.MODERATE: rng.uniform(5.0, 6.0),
                EventSeverity.MAJOR: rng.uniform(6.0, 7.5),
                EventSeverity.EXTREME: rng.uniform(7.5, 9.0)
            }[severity]
            magnitude_indicator = f"震级 {magnitude:.1f}"
        elif disaster_type in ["hurricane", "tornado"]:
            # 风速级别
            category = {
                EventSeverity.MINOR: 1,
                EventSeverity.MODERATE: 2,
                EventSeverity.MAJOR: 3,
                EventSeverity.EXTREME: rng.choice([4, 5])
            }[severity]
            magnitude_indicator = f"等级 {category}"
        else:
            magnitude_indicator = ""
        
        description = f"{region}发生{disaster_name}"
        if magnitude_indicator:
            description += f"（{magnitude_indicator}）"
        
        event = Event(
            id=f"disaster_{current_time.year}_{rng.randint(1000, 9999)}",
            type=EventType.NATURAL_DISASTER,
            subtype=disaster_type,
            name=f"{region}{disaster_name} ({current_time.year})",
            description=description,
            region=region,
            severity=severity,
            start_year=current_time.year,
            duration=duration,
            scope="regional",
            metadata={
                "disaster_type": disaster_type,
                "magnitude": magnitude if disaster_type == "earthquake" else None,
                "category": category if disaster_type in ["hurricane", "tornado"] else None
            }
        )
        
        return event
    
    def _generate_political_crisis(self, current_time: datetime,
                                 economic_state: Optional[Dict[str, Any]],
                                 political_state: Optional[Dict[str, Any]],
                                 technological_state: Optional[Dict[str, Any]],
                                 climate_state: Optional[Dict[str, Any]],
                                 rng: random.Random) -> Optional[Event]:
        """
        生成政治危机事件
        """
        crisis_types = [
            "government_collapse",
            "election_controversy",
            "corruption_scandal",
            "coup_attempt",
            "civil_unrest",
            "dictatorship_formation"
        ]
        
        crisis_type = rng.choice(crisis_types)
        country = rng.choice(self.major_countries + ["Iran", "Saudi_Arabia", "North_Korea", "Venezuela"])
        
        # 严重程度
        severity_roll = rng.random()
        if severity_roll < 0.15:
            severity = EventSeverity.EXTREME
        elif severity_roll < 0.4:
            severity = EventSeverity.MAJOR
        elif severity_roll < 0.7:
            severity = EventSeverity.MODERATE
        else:
            severity = EventSeverity.MINOR
        
        # 持续时间
        duration = {
            EventSeverity.MINOR: rng.randint(1, 2),
            EventSeverity.MODERATE: rng.randint(2, 4),
            EventSeverity.MAJOR: rng.randint(3, 6),
            EventSeverity.EXTREME: rng.randint(5, 10)
        }[severity]
        
        # 危机描述
        crisis_names = {
            "government_collapse": "政府倒台",
            "election_controversy": "选举争议",
            "corruption_scandal": "腐败丑闻",
            "coup_attempt": "政变企图",
            "civil_unrest": "国内动荡",
            "dictatorship_formation": "独裁政权形成"
        }
        
        crisis_name = crisis_names.get(crisis_type, "政治危机")
        
        event = Event(
            id=f"political_crisis_{current_time.year}_{rng.randint(1000, 9999)}",
            type=EventType.POLITICAL_CRISIS,
            subtype=crisis_type,
            name=f"{country}{crisis_name} ({current_time.year})",
            description=f"{country}发生{crisis_name}",
            region=self._get_region_for_country(country),
            severity=severity,
            start_year=current_time.year,
            duration=duration,
            scope="regional",
            metadata={
                "country": country,
                "crisis_type": crisis_type
            }
        )
        
        return event
    
    def _generate_resource_crisis(self, current_time: datetime,
                                economic_state: Optional[Dict[str, Any]],
                                political_state: Optional[Dict[str, Any]],
                                technological_state: Optional[Dict[str, Any]],
                                climate_state: Optional[Dict[str, Any]],
                                rng: random.Random) -> Optional[Event]:
        """
        生成资源危机事件
        """
        resource_types = [
            "oil_crisis",
            "water_shortage",
            "food_crisis",
            "rare_minerals_shortage",
            "energy_crisis"
        ]
        
        resource_type = rng.choice(resource_types)
        
        # 影响地区
        if rng.random() < 0.4:
            region = "Global"
        else:
            region = rng.choice(self.regions)
        
        # 严重程度
        severity_roll = rng.random()
        if severity_roll < 0.1:
            severity = EventSeverity.EXTREME
        elif severity_roll < 0.3:
            severity = EventSeverity.MAJOR
        elif severity_roll < 0.6:
            severity = EventSeverity.MODERATE
        else:
            severity = EventSeverity.MINOR
        
        # 持续时间
        duration = {
            EventSeverity.MINOR: rng.randint(1, 3),
            EventSeverity.MODERATE: rng.randint(2, 5),
            EventSeverity.MAJOR: rng.randint(4, 8),
            EventSeverity.EXTREME: rng.randint(6, 12)
        }[severity]
        
        # 资源描述
        resource_names = {
            "oil_crisis": "石油危机",
            "water_shortage": "水资源短缺",
            "food_crisis": "粮食危机",
            "rare_minerals_shortage": "稀有矿物短缺",
            "energy_crisis": "能源危机"
        }
        
        resource_name = resource_names.get(resource_type, "资源危机")
        
        # 价格影响
        price_increase = {
            EventSeverity.MINOR: rng.uniform(10, 30),
            EventSeverity.MODERATE: rng.uniform(30, 70),
            EventSeverity.MAJOR: rng.uniform(70, 150),
            EventSeverity.EXTREME: rng.uniform(150, 300)
        }[severity]
        
        event = Event(
            id=f"resource_crisis_{current_time.year}_{rng.randint(1000, 9999)}",
            type=EventType.RESOURCE_CRISIS,
            subtype=resource_type,
            name=f"{region}{resource_name} ({current_time.year})",
            description=f"{region}爆发{resource_name}",
            region=region,
            severity=severity,
            start_year=current_time.year,
            duration=duration,
            scope="global" if region == "Global" else "regional",
            metadata={
                "resource_type": resource_type,
                "price_increase_percentage": price_increase
            }
        )
        
        return event
    
    def _generate_social_unrest(self, current_time: datetime,
                              economic_state: Optional[Dict[str, Any]],
                              political_state: Optional[Dict[str, Any]],
                              technological_state: Optional[Dict[str, Any]],
                              climate_state: Optional[Dict[str, Any]],
                              rng: random.Random) -> Optional[Event]:
        """
        生成社会动荡事件
        """
        unrest_types = [
            "protests",
            "riots",
            "strikes",
            "civil_disobedience",
            "terrorist_attacks",
            "refugee_crisis"
        ]
        
        unrest_type = rng.choice(unrest_types)
        
        # 选择国家或地区
        if rng.random() < 0.3:
            # 地区性动荡
            region = rng.choice(self.regions)
            location = region
        else:
            # 国家性动荡
            country = rng.choice(self.major_countries + ["Iran", "Egypt", "Brazil", "South_Africa"])
            location = country
        
        # 严重程度
        severity_roll = rng.random()
        if severity_roll < 0.1:
            severity = EventSeverity.EXTREME
        elif severity_roll < 0.3:
            severity = EventSeverity.MAJOR
        elif severity_roll < 0.6:
            severity = EventSeverity.MODERATE
        else:
            severity = EventSeverity.MINOR
        
        # 持续时间
        duration = {
            EventSeverity.MINOR: rng.randint(1, 2),
            EventSeverity.MODERATE: rng.randint(2, 4),
            EventSeverity.MAJOR: rng.randint(3, 6),
            EventSeverity.EXTREME: rng.randint(5, 10)
        }[severity]
        
        # 动荡描述
        unrest_names = {
            "protests": "抗议活动",
            "riots": "暴动",
            "strikes": "罢工浪潮",
            "civil_disobedience": "公民不服从运动",
            "terrorist_attacks": "恐怖袭击",
            "refugee_crisis": "难民危机"
        }
        
        unrest_name = unrest_names.get(unrest_type, "社会动荡")
        
        # 参与者规模
        participant_scale = {
            EventSeverity.MINOR: "小规模",
            EventSeverity.MODERATE: "中规模",
            EventSeverity.MAJOR: "大规模",
            EventSeverity.EXTREME: "全国性"
        }[severity]
        
        event = Event(
            id=f"social_unrest_{current_time.year}_{rng.randint(1000, 9999)}",
            type=EventType.SOCIAL_UNREST,
            subtype=unrest_type,
            name=f"{location}{unrest_name} ({current_time.year})",
            description=f"{location}爆发{participant_scale}{unrest_name}",
            region=self._get_region_for_country(location) if location not in self.regions else location,
            severity=severity,
            start_year=current_time.year,
            duration=duration,
            scope="regional",
            metadata={
                "unrest_type": unrest_type,
                "location": location,
                "scale": participant_scale
            }
        )
        
        return event
    
    def _calculate_event_impacts(self, event: Event,
                               economic_state: Optional[Dict[str, Any]] = None,
                               political_state: Optional[Dict[str, Any]] = None,
                               technological_state: Optional[Dict[str, Any]] = None,
                               climate_state: Optional[Dict[str, Any]] = None) -> Event:
        """
        计算事件对各领域的影响
        
        Args:
            event: 事件对象
            economic_state: 经济状态
            political_state: 政治状态
            technological_state: 技术状态
            climate_state: 气候状态
            
        Returns:
            包含影响信息的事件对象
        """
        # 初始化影响字典
        event.impacts = {
            "economic": {},
            "political": {},
            "technological": {},
            "climate": {},
            "social": {}
        }
        
        # 调用对应的影响计算器
        if event.type in self.impact_calculators:
            self.impact_calculators[event.type](
                event, economic_state, political_state,
                technological_state, climate_state
            )
        
        return event
    
    def _calculate_economic_crisis_impact(self, event: Event,
                                        economic_state: Optional[Dict[str, Any]],
                                        political_state: Optional[Dict[str, Any]],
                                        technological_state: Optional[Dict[str, Any]],
                                        climate_state: Optional[Dict[str, Any]]):
        """
        计算经济危机的影响
        """
        # 严重程度系数
        severity_factor = {
            EventSeverity.MINOR: 0.2,
            EventSeverity.MODERATE: 0.5,
            EventSeverity.MAJOR: 0.8,
            EventSeverity.EXTREME: 1.0
        }[event.severity]
        
        # 范围系数
        scope_factor = 1.0 if event.scope == "global" else 0.3
        
        # GDP影响
        gdp_impact = -5.0 * severity_factor * scope_factor  # GDP下降百分比
        
        # 失业率影响
        unemployment_impact = 3.0 * severity_factor * scope_factor  # 失业率上升百分点
        
        # 通货膨胀影响（取决于危机类型）
        if event.subtype in ["inflation_crisis", "currency_crisis"]:
            inflation_impact = 8.0 * severity_factor * scope_factor  # 通胀率上升
        elif event.subtype in ["recession", "financial_crisis"]:
            inflation_impact = -2.0 * severity_factor * scope_factor  # 可能导致通缩
        else:
            inflation_impact = 2.0 * severity_factor * scope_factor
        
        # 贸易影响
        trade_impact = -10.0 * severity_factor * scope_factor  # 贸易下降百分比
        
        # 经济不确定性
        uncertainty_increase = 50.0 * severity_factor * scope_factor  # 不确定性指数上升
        
        # 设置影响
        event.impacts["economic"] = {
            "gdp_change": gdp_impact,
            "unemployment_change": unemployment_impact,
            "inflation_change": inflation_impact,
            "trade_change": trade_impact,
            "uncertainty_increase": uncertainty_increase,
            "duration_years": event.duration
        }
        
        # 政治影响
        political_instability = 20.0 * severity_factor * scope_factor  # 政治不稳定性上升
        event.impacts["political"] = {
            "instability_increase": political_instability
        }
        
        # 社会影响
        social_unrest_increase = 30.0 * severity_factor * scope_factor  # 社会动荡增加
        poverty_increase = 15.0 * severity_factor * scope_factor  # 贫困率上升
        event.impacts["social"] = {
            "unrest_increase": social_unrest_increase,
            "poverty_increase": poverty_increase
        }
    
    def _calculate_technological_breakthrough_impact(self, event: Event,
                                                   economic_state: Optional[Dict[str, Any]],
                                                   political_state: Optional[Dict[str, Any]],
                                                   technological_state: Optional[Dict[str, Any]],
                                                   climate_state: Optional[Dict[str, Any]]):
        """
        计算技术突破的影响
        """
        # 严重程度系数
        severity_factor = {
            EventSeverity.MINOR: 0.1,
            EventSeverity.MODERATE: 0.3,
            EventSeverity.MAJOR: 0.6,
            EventSeverity.EXTREME: 1.0
        }[event.severity]
        
        # 范围系数
        scope_factor = 1.0 if event.scope == "global" else 0.4
        
        # 经济影响（正面）
        gdp_growth = 2.0 * severity_factor * scope_factor  # GDP增长提升
        productivity_increase = 5.0 * severity_factor * scope_factor  # 生产力提升
        new_jobs = 3.0 * severity_factor * scope_factor  # 新增就业机会
        
        # 技术领域特定影响
        tech_impact = {
            "ai_advancements": 100.0 * severity_factor if event.subtype == "artificial_intelligence" else 0,
            "renewable_efficiency": 50.0 * severity_factor if event.subtype == "renewable_energy" else 0,
            "healthcare_improvement": 40.0 * severity_factor if event.subtype == "biotechnology" else 0,
            "computing_power": 80.0 * severity_factor if event.subtype == "quantum_computing" else 0,
            "clean_energy": 60.0 * severity_factor if event.subtype == "fusion_energy" else 0
        }
        
        # 气候变化影响（某些技术可能有正面影响）
        carbon_reduction = 0
        if event.subtype in ["renewable_energy", "fusion_energy", "advanced_materials"]:
            carbon_reduction = 10.0 * severity_factor * scope_factor  # 碳减排潜力
        
        # 设置影响
        event.impacts["economic"] = {
            "gdp_growth_potential": gdp_growth,
            "productivity_increase": productivity_increase,
            "new_jobs_potential": new_jobs,
            "commercialization_delay_years": event.metadata.get("commercialization_time", 5)
        }
        
        event.impacts["technological"] = {
            "field_advancement": tech_impact,
            "rd_investment_increase": 30.0 * severity_factor,
            "knowledge_sharing": 20.0 * severity_factor * scope_factor
        }
        
        if carbon_reduction > 0:
            event.impacts["climate"] = {
                "carbon_reduction_potential": carbon_reduction
            }
        
        # 社会影响
        event.impacts["social"] = {
            "lifestyle_changes": 20.0 * severity_factor * scope_factor,
            "education_demand": 15.0 * severity_factor
        }
    
    def _calculate_conflict_impact(self, event: Event,
                                 economic_state: Optional[Dict[str, Any]],
                                 political_state: Optional[Dict[str, Any]],
                                 technological_state: Optional[Dict[str, Any]],
                                 climate_state: Optional[Dict[str, Any]]):
        """
        计算冲突的影响
        """
        # 严重程度系数
        severity_factor = {
            EventSeverity.MINOR: 0.1,
            EventSeverity.MODERATE: 0.3,
            EventSeverity.MAJOR: 0.7,
            EventSeverity.EXTREME: 1.0
        }[event.severity]
        
        # 冲突类型系数
        type_factor = {
            "military_conflict": 1.0,
            "trade_war": 0.6,
            "cyber_conflict": 0.7,
            "proxy_war": 0.8,
            "border_dispute": 0.5
        }.get(event.subtype, 0.7)
        
        # 经济影响
        gdp_impact = -8.0 * severity_factor * type_factor  # GDP下降
        trade_impact = -15.0 * severity_factor * type_factor  # 贸易下降
        inflation_impact = 5.0 * severity_factor * type_factor  # 通胀上升
        
        # 政治影响
        political_instability = 40.0 * severity_factor * type_factor  # 政治不稳定性
        alliance_changes = 30.0 * severity_factor * type_factor  # 联盟变化
        
        # 人口影响（军事冲突）
        casualty_rate = 0
        displacement = 0
        if event.subtype in ["military_conflict", "proxy_war"]:
            casualty_rate = 0.5 * severity_factor  # 伤亡率（百分比）
            displacement = 5.0 * severity_factor  # 流离失所人口（百分比）
        
        # 技术影响（可能促进军事技术发展）
        military_tech_advance = 10.0 * severity_factor * type_factor
        
        # 气候变化影响（可能有环境破坏）
        environmental_damage = 0
        if event.subtype in ["military_conflict", "proxy_war"]:
            environmental_damage = 20.0 * severity_factor  # 环境破坏指数
        
        # 设置影响
        event.impacts["economic"] = {
            "gdp_change": gdp_impact,
            "trade_change": trade_impact,
            "inflation_change": inflation_impact,
            "defense_spending_increase": 30.0 * severity_factor
        }
        
        event.impacts["political"] = {
            "instability_increase": political_instability,
            "alliance_changes": alliance_changes,
            "participants": event.metadata.get("participants", [])
        }
        
        event.impacts["technological"] = {
            "military_advancements": military_tech_advance
        }
        
        event.impacts["social"] = {
            "casualty_rate": casualty_rate,
            "displacement_rate": displacement,
            "refugee_crisis": 15.0 * severity_factor if displacement > 0 else 0
        }
        
        if environmental_damage > 0:
            event.impacts["climate"] = {
                "environmental_damage": environmental_damage
            }
    
    def _calculate_pandemic_impact(self, event: Event,
                                 economic_state: Optional[Dict[str, Any]],
                                 political_state: Optional[Dict[str, Any]],
                                 technological_state: Optional[Dict[str, Any]],
                                 climate_state: Optional[Dict[str, Any]]):
        """
        计算大流行病的影响
        """
        # 严重程度系数
        severity_factor = {
            EventSeverity.MINOR: 0.1,
            EventSeverity.MODERATE: 0.3,
            EventSeverity.MAJOR: 0.7,
            EventSeverity.EXTREME: 1.0
        }[event.severity]
        
        # 范围系数
        scope_factor = 1.0 if event.scope == "global" else 0.4
        
        # 经济影响
        gdp_impact = -10.0 * severity_factor * scope_factor  # GDP下降
        unemployment_impact = 8.0 * severity_factor * scope_factor  # 失业率上升
        healthcare_spending = 40.0 * severity_factor * scope_factor  # 医疗支出增加
        
        # 社会影响
        mortality = event.metadata.get("mortality_rate", 1.0) * severity_factor
        infection_rate = event.metadata.get("transmission_rate", 3.0)
        social_distancing = 70.0 * severity_factor * scope_factor  # 社会隔离程度
        
        # 技术影响（促进医疗技术发展）
        medical_tech_advance = 30.0 * severity_factor  # 医疗技术进步
        biotech_investment = 50.0 * severity_factor * scope_factor  # 生物技术投资增加
        
        # 设置影响
        event.impacts["economic"] = {
            "gdp_change": gdp_impact,
            "unemployment_change": unemployment_impact,
            "healthcare_spending_increase": healthcare_spending,
            "service_industry_impact": -50.0 * severity_factor * scope_factor
        }
        
        event.impacts["technological"] = {
            "medical_advancements": medical_tech_advance,
            "biotech_investment": biotech_investment,
            "remote_work_adoption": 60.0 * severity_factor * scope_factor
        }
        
        event.impacts["social"] = {
            "mortality_rate": mortality,
            "infection_rate": infection_rate,
            "social_disruption": 80.0 * severity_factor * scope_factor,
            "mental_health_impact": 40.0 * severity_factor * scope_factor
        }
    
    def _calculate_natural_disaster_impact(self, event: Event,
                                         economic_state: Optional[Dict[str, Any]],
                                         political_state: Optional[Dict[str, Any]],
                                         technological_state: Optional[Dict[str, Any]],
                                         climate_state: Optional[Dict[str, Any]]):
        """
        计算自然灾害的影响
        """
        # 严重程度系数
        severity_factor = {
            EventSeverity.MINOR: 0.1,
            EventSeverity.MODERATE: 0.3,
            EventSeverity.MAJOR: 0.7,
            EventSeverity.EXTREME: 1.0
        }[event.severity]
        
        # 灾害类型系数
        type_factor = {
            "earthquake": 1.0,
            "tsunami": 0.9,
            "hurricane": 0.8,
            "tornado": 0.5,
            "flood": 0.7,
            "drought": 0.8,
            "wildfire": 0.6,
            "volcanic_eruption": 0.8,
            "heat_wave": 0.5,
            "cold_wave": 0.4
        }.get(event.subtype, 0.7)
        
        # 经济影响
        gdp_impact = -6.0 * severity_factor * type_factor  # GDP下降
        infrastructure_damage = 70.0 * severity_factor * type_factor  # 基础设施破坏
        recovery_cost = 50.0 * severity_factor * type_factor  # 恢复成本（占GDP百分比）
        
        # 社会影响
        displacement = 10.0 * severity_factor * type_factor  # 流离失所人口
        mortality = 0.2 * severity_factor * type_factor  # 死亡率（百分比）
        
        # 环境影响
        environmental_damage = 30.0 * severity_factor * type_factor  # 环境破坏
        
        # 技术影响（可能促进防灾技术）
        disaster_tech_advance = 15.0 * severity_factor  # 防灾技术进步
        
        # 设置影响
        event.impacts["economic"] = {
            "gdp_change": gdp_impact,
            "infrastructure_damage": infrastructure_damage,
            "recovery_cost": recovery_cost,
            "insurance_claims": 60.0 * severity_factor * type_factor
        }
        
        event.impacts["social"] = {
            "displacement_rate": displacement,
            "mortality_rate": mortality,
            "humanitarian_crisis": 40.0 * severity_factor * type_factor
        }
        
        event.impacts["climate"] = {
            "environmental_damage": environmental_damage,
            "emissions_increase": 10.0 * severity_factor if event.subtype in ["wildfire", "volcanic_eruption"] else 0
        }
        
        event.impacts["technological"] = {
            "disaster_prevention_tech": disaster_tech_advance
        }
    
    def _calculate_political_crisis_impact(self, event: Event,
                                         economic_state: Optional[Dict[str, Any]],
                                         political_state: Optional[Dict[str, Any]],
                                         technological_state: Optional[Dict[str, Any]],
                                         climate_state: Optional[Dict[str, Any]]):
        """
        计算政治危机的影响
        """
        # 严重程度系数
        severity_factor = {
            EventSeverity.MINOR: 0.1,
            EventSeverity.MODERATE: 0.3,
            EventSeverity.MAJOR: 0.7,
            EventSeverity.EXTREME: 1.0
        }[event.severity]
        
        # 危机类型系数
        type_factor = {
            "government_collapse": 1.0,
            "election_controversy": 0.6,
            "corruption_scandal": 0.5,
            "coup_attempt": 0.9,
            "civil_unrest": 0.7,
            "dictatorship_formation": 0.8
        }.get(event.subtype, 0.6)
        
        # 经济影响
        gdp_impact = -4.0 * severity_factor * type_factor  # GDP下降
        investment_impact = -20.0 * severity_factor * type_factor  # 投资减少
        currency_devaluation = 15.0 * severity_factor * type_factor  # 货币贬值
        
        # 政治影响
        regional_instability = 30.0 * severity_factor * type_factor  # 区域不稳定性
        regime_change_probability = 50.0 * severity_factor * type_factor if event.subtype in ["government_collapse", "coup_attempt"] else 0
        
        # 社会影响
        social_unrest = 40.0 * severity_factor * type_factor  # 社会动荡
        human_rights_impact = 35.0 * severity_factor * type_factor if event.subtype == "dictatorship_formation" else 20.0 * severity_factor * type_factor
        
        # 设置影响
        event.impacts["economic"] = {
            "gdp_change": gdp_impact,
            "investment_change": investment_impact,
            "currency_devaluation": currency_devaluation,
            "trade_partner_impact": -15.0 * severity_factor * type_factor
        }
        
        event.impacts["political"] = {
            "instability_increase": regional_instability,
            "regime_change_probability": regime_change_probability,
            "affected_country": event.metadata.get("country", "")
        }
        
        event.impacts["social"] = {
            "unrest_increase": social_unrest,
            "human_rights_impact": human_rights_impact,
            "media_freedom_impact": -25.0 * severity_factor * type_factor
        }
    
    def _calculate_resource_crisis_impact(self, event: Event,
                                        economic_state: Optional[Dict[str, Any]],
                                        political_state: Optional[Dict[str, Any]],
                                        technological_state: Optional[Dict[str, Any]],
                                        climate_state: Optional[Dict[str, Any]]):
        """
        计算资源危机的影响
        """
        # 严重程度系数
        severity_factor = {
            EventSeverity.MINOR: 0.1,
            EventSeverity.MODERATE: 0.3,
            EventSeverity.MAJOR: 0.7,
            EventSeverity.EXTREME: 1.0
        }[event.severity]
        
        # 范围系数
        scope_factor = 1.0 if event.scope == "global" else 0.4
        
        # 资源类型系数
        type_factor = {
            "oil_crisis": 1.0,
            "water_shortage": 0.9,
            "food_crisis": 0.9,
            "rare_minerals_shortage": 0.6,
            "energy_crisis": 0.8
        }.get(event.subtype, 0.7)
        
        # 经济影响
        price_increase = event.metadata.get("price_increase_percentage", 50) * severity_factor
        gdp_impact = -5.0 * severity_factor * type_factor * scope_factor  # GDP下降
        inflation_impact = 7.0 * severity_factor * type_factor * scope_factor  # 通胀上升
        
        # 政治影响（资源争夺）
        geopolitical_tensions = 25.0 * severity_factor * type_factor * scope_factor  # 地缘政治紧张
        
        # 社会影响
        social_unrest = 35.0 * severity_factor * type_factor * scope_factor  # 社会动荡
        
        # 技术影响（促进替代技术）
        alternative_tech_investment = 30.0 * severity_factor * type_factor  # 替代技术投资
        efficiency_improvement = 25.0 * severity_factor * type_factor  # 效率提升
        
        # 气候变化影响（某些危机可能促使低碳转型）
        carbon_reduction_potential = 0
        if event.subtype in ["oil_crisis", "energy_crisis"]:
            carbon_reduction_potential = 15.0 * severity_factor * scope_factor
        
        # 设置影响
        event.impacts["economic"] = {
            "gdp_change": gdp_impact,
            "inflation_change": inflation_impact,
            "commodity_price_increase": price_increase,
            "industry_disruption": 40.0 * severity_factor * type_factor * scope_factor
        }
        
        event.impacts["political"] = {
            "geopolitical_tensions": geopolitical_tensions,
            "trade_disputes": 20.0 * severity_factor * type_factor * scope_factor
        }
        
        event.impacts["social"] = {
            "unrest_increase": social_unrest,
            "access_inequality": 30.0 * severity_factor * type_factor * scope_factor
        }
        
        event.impacts["technological"] = {
            "alternative_tech_investment": alternative_tech_investment,
            "efficiency_improvements": efficiency_improvement,
            "conservation_tech": 25.0 * severity_factor * type_factor
        }
        
        if carbon_reduction_potential > 0:
            event.impacts["climate"] = {
                "carbon_reduction_potential": carbon_reduction_potential
            }
    
    def _calculate_social_unrest_impact(self, event: Event,
                                      economic_state: Optional[Dict[str, Any]],
                                      political_state: Optional[Dict[str, Any]],
                                      technological_state: Optional[Dict[str, Any]],
                                      climate_state: Optional[Dict[str, Any]]):
        """
        计算社会动荡的影响
        """
        # 严重程度系数
        severity_factor = {
            EventSeverity.MINOR: 0.1,
            EventSeverity.MODERATE: 0.3,
            EventSeverity.MAJOR: 0.7,
            EventSeverity.EXTREME: 1.0
        }[event.severity]
        
        # 动荡类型系数
        type_factor = {
            "protests": 0.5,
            "riots": 0.9,
            "strikes": 0.6,
            "civil_disobedience": 0.4,
            "terrorist_attacks": 1.0,
            "refugee_crisis": 0.8
        }.get(event.subtype, 0.6)
        
        # 经济影响
        gdp_impact = -3.0 * severity_factor * type_factor  # GDP下降
        business_disruption = 40.0 * severity_factor * type_factor  # 商业中断
        tourism_impact = -50.0 * severity_factor * type_factor if event.subtype == "terrorist_attacks" else -20.0 * severity_factor * type_factor
        
        # 政治影响
        government_legitimacy = -30.0 * severity_factor * type_factor  # 政府合法性下降
        policy_changes = 40.0 * severity_factor * type_factor  # 政策变化概率
        
        # 社会影响
        security_increase = 50.0 * severity_factor * type_factor  # 安全措施增加
        polarization = 35.0 * severity_factor * type_factor  # 社会极化
        
        # 设置影响
        event.impacts["economic"] = {
            "gdp_change": gdp_impact,
            "business_disruption": business_disruption,
            "tourism_impact": tourism_impact,
            "security_spending": 30.0 * severity_factor * type_factor
        }
        
        event.impacts["political"] = {
            "government_legitimacy": government_legitimacy,
            "policy_change_probability": policy_changes,
            "affected_region": event.metadata.get("location", "")
        }
        
        event.impacts["social"] = {
            "security_measures_increase": security_increase,
            "social_polarization": polarization,
            "trust_decline": 25.0 * severity_factor * type_factor
        }
    
    def _get_region_for_country(self, country: str) -> str:
        """
        获取国家所属的地区
        
        Args:
            country: 国家名称
            
        Returns:
            地区名称
        """
        country_regions = {
            "United_States": "North_America",
            "Canada": "North_America",
            "Mexico": "North_America",
            "Brazil": "South_America",
            "Argentina": "South_America",
            "Venezuela": "South_America",
            "Colombia": "South_America",
            "Germany": "Europe",
            "France": "Europe",
            "United_Kingdom": "Europe",
            "Russia": "Europe",
            "Ukraine": "Europe",
            "China": "Asia",
            "India": "Asia",
            "Japan": "Asia",
            "South_Korea": "Asia",
            "North_Korea": "Asia",
            "Pakistan": "Asia",
            "Iran": "Middle_East",
            "Saudi_Arabia": "Middle_East",
            "Israel": "Middle_East",
            "Turkey": "Middle_East",
            "Egypt": "Africa",
            "South_Africa": "Africa",
            "Nigeria": "Africa",
            "Australia": "Oceania",
            "New_Zealand": "Oceania"
        }
        
        return country_regions.get(country, "Unknown")
    
    def _get_region_for_countries(self, country1: str, country2: str) -> str:
        """
        获取多个国家共同所属的地区，如果在不同地区则返回"Multiple"
        
        Args:
            country1: 第一个国家
            country2: 第二个国家
            
        Returns:
            地区名称
        """
        region1 = self._get_region_for_country(country1)
        region2 = self._get_region_for_country(country2)
        
        if region1 == region2:
            return region1
        else:
            return "Multiple"
    
    def get_historical_events(self, start_year: Optional[int] = None,
                            end_year: Optional[int] = None,
                            event_types: Optional[List[EventType]] = None,
                            regions: Optional[List[str]] = None) -> List[Event]:
        """
        获取历史事件
        
        Args:
            start_year: 开始年份
            end_year: 结束年份
            event_types: 事件类型列表
            regions: 地区列表
            
        Returns:
            符合条件的历史事件列表
        """
        filtered_events = self.event_history.copy()
        
        # 按年份过滤
        if start_year is not None:
            filtered_events = [e for e in filtered_events if e.start_year >= start_year]
        
        if end_year is not None:
            filtered_events = [e for e in filtered_events if e.start_year <= end_year]
        
        # 按事件类型过滤
        if event_types is not None:
            filtered_events = [e for e in filtered_events if e.type in event_types]
        
        # 按地区过滤
        if regions is not None:
            filtered_events = [e for e in filtered_events if e.region in regions]
        
        return filtered_events
    
    def clear_history(self):
        """
        清除事件历史记录
        """
        self.event_history.clear()
        self.logger.info("事件历史记录已清除")
    
    def calculate_global_risk_index(self) -> float:
        """
        计算全球风险指数
        
        Returns:
            全球风险指数（0-100）
        """
        # 获取最近5年的事件
        current_year = datetime.now().year
        recent_events = self.get_historical_events(start_year=current_year - 5)
        
        # 计算风险指数
        risk_score = 0
        for event in recent_events:
            # 严重程度权重
            severity_weight = {
                EventSeverity.MINOR: 1,
                EventSeverity.MODERATE: 2,
                EventSeverity.MAJOR: 4,
                EventSeverity.EXTREME: 8
            }[event.severity]
            
            # 范围权重
            scope_weight = 2 if event.scope == "global" else 1
            
            # 时间衰减
            year_diff = current_year - event.start_year
            time_decay = 1.0 / (1.0 + year_diff * 0.2)
            
            # 事件特定权重
            event_weight = {
                EventType.ECONOMIC_CRISIS: 0.8,
                EventType.TECHNOLOGICAL_BREAKTHROUGH: -0.3,  # 技术突破可能降低风险
                EventType.CONFLICT: 1.0,
                EventType.PANDEMIC: 0.9,
                EventType.NATURAL_DISASTER: 0.7,
                EventType.POLITICAL_CRISIS: 0.8,
                EventType.RESOURCE_CRISIS: 0.8,
                EventType.SOCIAL_UNREST: 0.7
            }.get(event.type, 0.8)
            
            # 计算事件贡献
            event_contribution = severity_weight * scope_weight * time_decay * event_weight
            risk_score += event_contribution
        
        # 归一化到0-100范围
        risk_index = min(max(risk_score, 0), 100)
        
        return risk_index
    
    def predict_event_probabilities(self, lookahead_years: int = 1,
                                  economic_state: Optional[Dict[str, Any]] = None,
                                  political_state: Optional[Dict[str, Any]] = None,
                                  technological_state: Optional[Dict[str, Any]] = None,
                                  climate_state: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        预测未来事件概率
        
        Args:
            lookahead_years: 预测年限
            economic_state: 经济状态
            political_state: 政治状态
            technological_state: 技术状态
            climate_state: 气候状态
            
        Returns:
            各事件类型的预测概率
        """
        predictions = {}
        current_time = datetime.now()
        
        # 对每种事件类型计算预测概率
        for event_type, base_prob in self.event_probabilities.items():
            # 调整概率
            adjusted_prob = self._adjust_event_probability(
                event_type, base_prob, economic_state, political_state,
                technological_state, climate_state, current_time
            )
            
            # 考虑预测年限
            yearly_prob = 1 - math.exp(-adjusted_prob)  # 转换为年度概率
            cumulative_prob = 1 - math.exp(-yearly_prob * lookahead_years)
            
            predictions[event_type] = cumulative_prob
        
        return predictions


# 测试代码
if __name__ == "__main__":
    # 配置
    config = {
        "event_probabilities": {
            "economic_crisis": 0.05,
            "technological_breakthrough": 0.08,
            "conflict": 0.06,
            "pandemic": 0.02,
            "natural_disaster": 0.12,
            "political_crisis": 0.07,
            "resource_crisis": 0.04,
            "social_unrest": 0.09
        },
        "max_events_per_year": 5
    }
    
    # 创建事件系统
    event_system = EventSystem(config)
    
    # 模拟状态
    mock_economic_state = {
        "global_gdp_growth": 3.2,
        "global_debt_level": 250,
        "global_inequality": 0.45,
        "energy_prices": {"oil": 75}
    }
    
    mock_political_state = {
        "global_stability": 0.65,
        "countries": {
            "United_States": {"stability": 0.7},
            "China": {"stability": 0.8},
            "Russia": {"stability": 0.5}
        }
    }
    
    mock_technological_state = {
        "global_rd_investment": 2.8,
        "technologies": {
            "artificial_intelligence": {"maturity": 0.75},
            "renewable_energy": {"maturity": 0.6},
            "quantum_computing": {"maturity": 0.4}
        }
    }
    
    mock_climate_state = {
        "temperature": {"global_mean": 1.2},
        "co2_level": 420
    }
    
    # 生成事件
    print("生成未来1年的事件...")
    events = event_system.generate_events(
        datetime.now(),
        mock_economic_state,
        mock_political_state,
        mock_technological_state,
        mock_climate_state,
        seed=42  # 固定种子以重现结果
    )
    
    # 打印生成的事件
    print(f"\n生成了 {len(events)} 个事件:")
    for i, event in enumerate(events, 1):
        print(f"\n事件 {i}:")
        print(f"  ID: {event.id}")
        print(f"  类型: {event.type}")
        print(f"  子类型: {event.subtype}")
        print(f"  名称: {event.name}")
        print(f"  描述: {event.description}")
        print(f"  地区: {event.region}")
        print(f"  严重程度: {event.severity}")
        print(f"  开始年份: {event.start_year}")
        print(f"  持续时间: {event.duration} 年")
        print(f"  范围: {event.scope}")
        
        # 打印影响
        print("  影响:")
        if "economic" in event.impacts and event.impacts["economic"]:
            print("    经济影响:")
            for key, value in event.impacts["economic"].items():
                print(f"      {key}: {value}")
        
        if "political" in event.impacts and event.impacts["political"]:
            print("    政治影响:")
            for key, value in event.impacts["political"].items():
                print(f"      {key}: {value}")
        
        if "social" in event.impacts and event.impacts["social"]:
            print("    社会影响:")
            for key, value in event.impacts["social"].items():
                print(f"      {key}: {value}")
    
    # 预测未来概率
    print("\n预测未来5年的事件概率:")
    predictions = event_system.predict_event_probabilities(
        lookahead_years=5,
        economic_state=mock_economic_state,
        political_state=mock_political_state,
        technological_state=mock_technological_state,
        climate_state=mock_climate_state
    )
    
    for event_type, probability in predictions.items():
        print(f"  {event_type}: {probability:.2%}")
    
    # 计算全球风险指数
    risk_index = event_system.calculate_global_risk_index()
    print(f"\n全球风险指数: {risk_index:.1f}/100")