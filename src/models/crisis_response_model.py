#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
危机响应模型模块
负责模拟各国和国际组织如何应对全球性危机事件
"""

import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import math

from ..utils.logger import get_logger
from ..utils.random_utils import get_seeded_random
from ..data.data_types import Event, EventType, EventSeverity, Country, ResponseType, ResponseEffectiveness


class CrisisResponseModel:
    """
    全球危机响应模型
    模拟各国和国际组织如何应对全球性危机事件
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化危机响应模型
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("CrisisResponseModel")
        self.config = config
        
        # 国家能力配置
        self.country_capabilities = config.get("country_capabilities", {
            "United_States": {
                "economic_power": 0.9,
                "military_power": 0.95,
                "soft_power": 0.8,
                "resilience": 0.75,
                "cooperation_willingness": 0.6,
                "crisis_experience": 0.85
            },
            "China": {
                "economic_power": 0.85,
                "military_power": 0.8,
                "soft_power": 0.6,
                "resilience": 0.85,
                "cooperation_willingness": 0.55,
                "crisis_experience": 0.8
            },
            "Russia": {
                "economic_power": 0.5,
                "military_power": 0.75,
                "soft_power": 0.45,
                "resilience": 0.7,
                "cooperation_willingness": 0.3,
                "crisis_experience": 0.75
            },
            "European_Union": {
                "economic_power": 0.8,
                "military_power": 0.65,
                "soft_power": 0.85,
                "resilience": 0.7,
                "cooperation_willingness": 0.8,
                "crisis_experience": 0.7
            },
            "India": {
                "economic_power": 0.65,
                "military_power": 0.65,
                "soft_power": 0.55,
                "resilience": 0.75,
                "cooperation_willingness": 0.65,
                "crisis_experience": 0.7
            },
            "Japan": {
                "economic_power": 0.7,
                "military_power": 0.55,
                "soft_power": 0.75,
                "resilience": 0.8,
                "cooperation_willingness": 0.7,
                "crisis_experience": 0.85
            },
            "Brazil": {
                "economic_power": 0.55,
                "military_power": 0.5,
                "soft_power": 0.5,
                "resilience": 0.65,
                "cooperation_willingness": 0.6,
                "crisis_experience": 0.6
            },
            "Canada": {
                "economic_power": 0.6,
                "military_power": 0.4,
                "soft_power": 0.7,
                "resilience": 0.75,
                "cooperation_willingness": 0.85,
                "crisis_experience": 0.6
            },
            "Australia": {
                "economic_power": 0.55,
                "military_power": 0.45,
                "soft_power": 0.65,
                "resilience": 0.7,
                "cooperation_willingness": 0.75,
                "crisis_experience": 0.65
            },
            "International_Organizations": {
                "coordination_capability": 0.7,
                "resource_mobilization": 0.6,
                "legitimacy": 0.75,
                "response_speed": 0.5,
                "crisis_experience": 0.8
            }
        })
        
        # 国际组织定义
        self.international_organizations = [
            "UN",          # 联合国
            "WHO",         # 世界卫生组织
            "IMF",         # 国际货币基金组织
            "World_Bank",  # 世界银行
            "WTO",         # 世界贸易组织
            "NATO",        # 北大西洋公约组织
            "EU",          # 欧盟
            "ASEAN",       # 东南亚国家联盟
            "African_Union", # 非洲联盟
            "G20"          # 二十国集团
        ]
        
        # 响应策略库
        self.response_strategies = {
            EventType.ECONOMIC_CRISIS: self._generate_economic_crisis_responses,
            EventType.TECHNOLOGICAL_BREAKTHROUGH: self._generate_technological_breakthrough_responses,
            EventType.CONFLICT: self._generate_conflict_responses,
            EventType.PANDEMIC: self._generate_pandemic_responses,
            EventType.NATURAL_DISASTER: self._generate_natural_disaster_responses,
            EventType.POLITICAL_CRISIS: self._generate_political_crisis_responses,
            EventType.RESOURCE_CRISIS: self._generate_resource_crisis_responses,
            EventType.SOCIAL_UNREST: self._generate_social_unrest_responses
        }
        
        # 历史响应记录
        self.response_history = []
        
        # 国家关系矩阵
        self.country_relations = self._initialize_country_relations()
        
        self.logger.info("危机响应模型初始化完成")
    
    def _initialize_country_relations(self) -> Dict[Tuple[str, str], float]:
        """
        初始化国家关系矩阵
        
        Returns:
            国家间关系强度字典（-1.0 到 1.0）
        """
        relations = {}
        major_countries = list(self.country_capabilities.keys())
        major_countries = [c for c in major_countries if c != "International_Organizations"]
        
        # 默认关系值
        for i, country1 in enumerate(major_countries):
            for country2 in major_countries[i:]:
                if country1 == country2:
                    relations[(country1, country2)] = 1.0  # 与自身关系
                    relations[(country2, country1)] = 1.0
                else:
                    # 随机生成基础关系值，倾向于中立
                    base_relation = random.uniform(-0.3, 0.5)
                    relations[(country1, country2)] = base_relation
                    relations[(country2, country1)] = base_relation
        
        # 设置一些特定关系
        relations[('United_States', 'European_Union')] = 0.8
        relations[('European_Union', 'United_States')] = 0.8
        
        relations[('United_States', 'China')] = -0.2
        relations[('China', 'United_States')] = -0.2
        
        relations[('United_States', 'Russia')] = -0.4
        relations[('Russia', 'United_States')] = -0.4
        
        relations[('China', 'Russia')] = 0.6
        relations[('Russia', 'China')] = 0.6
        
        relations[('European_Union', 'Russia')] = -0.3
        relations[('Russia', 'European_Union')] = -0.3
        
        relations[('United_States', 'Japan')] = 0.75
        relations[('Japan', 'United_States')] = 0.75
        
        relations[('China', 'Japan')] = -0.3
        relations[('Japan', 'China')] = -0.3
        
        return relations
    
    def generate_responses(self, event: Event,
                         world_state: Dict[str, Any],
                         political_state: Optional[Dict[str, Any]] = None,
                         economic_state: Optional[Dict[str, Any]] = None,
                         seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        生成对事件的响应
        
        Args:
            event: 事件对象
            world_state: 世界状态
            political_state: 政治状态
            economic_state: 经济状态
            seed: 随机种子
            
        Returns:
            响应列表
        """
        try:
            # 初始化随机数生成器
            if seed is not None:
                rng = random.Random(seed)
            else:
                rng = get_seeded_random(hash(str(event.id)))
            
            responses = []
            
            # 检查是否有对应的响应策略生成器
            if event.type in self.response_strategies:
                # 生成国家响应
                country_responses = self._generate_country_responses(
                    event, world_state, political_state, economic_state, rng
                )
                responses.extend(country_responses)
                
                # 生成国际组织响应
                org_responses = self._generate_organization_responses(
                    event, world_state, political_state, economic_state, rng
                )
                responses.extend(org_responses)
                
                # 记录响应
                for response in responses:
                    response_record = {
                        "event_id": event.id,
                        "response": response,
                        "timestamp": datetime.now()
                    }
                    self.response_history.append(response_record)
            
            self.logger.info(f"为事件 {event.id} 生成了 {len(responses)} 个响应")
            return responses
            
        except Exception as e:
            self.logger.error(f"生成响应失败: {str(e)}")
            return []
    
    def _generate_country_responses(self, event: Event,
                                  world_state: Dict[str, Any],
                                  political_state: Optional[Dict[str, Any]] = None,
                                  economic_state: Optional[Dict[str, Any]] = None,
                                  rng: Optional[random.Random] = None) -> List[Dict[str, Any]]:
        """
        生成国家响应
        
        Args:
            event: 事件对象
            world_state: 世界状态
            political_state: 政治状态
            economic_state: 经济状态
            rng: 随机数生成器
            
        Returns:
            国家响应列表
        """
        if rng is None:
            rng = random.Random()
        
        responses = []
        
        # 确定受影响的国家
        affected_countries = self._identify_affected_countries(event, world_state)
        
        # 确定主要大国
        major_countries = [c for c in self.country_capabilities.keys() 
                          if c != "International_Organizations"]
        
        # 生成受影响国家的响应
        for country in affected_countries:
            if country in major_countries:
                response = self._generate_single_country_response(
                    country, event, world_state, political_state, economic_state, rng,
                    is_directly_affected=True
                )
                if response:
                    responses.append(response)
        
        # 生成其他主要国家的响应
        for country in major_countries:
            if country not in affected_countries:
                # 决定是否响应
                response_probability = self._calculate_response_probability(
                    country, event, world_state, political_state
                )
                
                if rng.random() < response_probability:
                    response = self._generate_single_country_response(
                        country, event, world_state, political_state, economic_state, rng,
                        is_directly_affected=False
                    )
                    if response:
                        responses.append(response)
        
        # 限制每个事件的国家响应数量
        max_responses = self.config.get("max_country_responses_per_event", 10)
        if len(responses) > max_responses:
            # 按响应重要性排序
            responses.sort(key=lambda r: r.get("importance", 0), reverse=True)
            responses = responses[:max_responses]
        
        return responses
    
    def _generate_single_country_response(self, country: str,
                                        event: Event,
                                        world_state: Dict[str, Any],
                                        political_state: Optional[Dict[str, Any]] = None,
                                        economic_state: Optional[Dict[str, Any]] = None,
                                        rng: Optional[random.Random] = None,
                                        is_directly_affected: bool = False) -> Optional[Dict[str, Any]]:
        """
        生成单个国家的响应
        
        Args:
            country: 国家名称
            event: 事件对象
            world_state: 世界状态
            political_state: 政治状态
            economic_state: 经济状态
            rng: 随机数生成器
            is_directly_affected: 是否直接受影响
            
        Returns:
            国家响应字典
        """
        if rng is None:
            rng = random.Random()
        
        # 获取国家能力
        capabilities = self.country_capabilities.get(country, {
            "economic_power": 0.5,
            "military_power": 0.5,
            "soft_power": 0.5,
            "resilience": 0.5,
            "cooperation_willingness": 0.5,
            "crisis_experience": 0.5
        })
        
        # 确定响应类型和强度
        response_types = self._get_appropriate_response_types(event)
        
        if not response_types:
            return None
        
        # 选择响应类型
        response_type = rng.choice(response_types)
        
        # 计算响应强度
        base_intensity = self._calculate_response_intensity(
            country, event, is_directly_affected, capabilities
        )
        
        # 调整响应强度
        intensity = min(max(base_intensity + rng.uniform(-0.1, 0.1), 0.1), 1.0)
        
        # 计算响应及时性（0-1）
        timeliness = self._calculate_response_timeliness(
            country, event, capabilities
        )
        
        # 计算响应有效性
        effectiveness = self._calculate_response_effectiveness(
            country, event, response_type, intensity, capabilities, rng
        )
        
        # 确定响应资源投入
        resources = self._calculate_resource_allocation(
            country, event, intensity, capabilities, economic_state
        )
        
        # 生成响应描述
        response_description = self._generate_response_description(
            country, event, response_type, intensity
        )
        
        # 计算国际合作程度
        cooperation_level = 0
        if response_type in [ResponseType.COOPERATION, ResponseType.HUMANITARIAN_AID]:
            cooperation_level = min(capabilities["cooperation_willingness"] + rng.uniform(-0.1, 0.1), 1.0)
        
        # 计算国内支持度
        public_support = self._calculate_public_support(
            country, event, response_type, intensity, rng
        )
        
        # 评估响应风险
        risks = self._assess_response_risks(
            country, event, response_type, intensity
        )
        
        response = {
            "actor_type": "country",
            "actor": country,
            "event_id": event.id,
            "response_type": response_type,
            "description": response_description,
            "intensity": intensity,
            "timeliness": timeliness,
            "effectiveness": effectiveness,
            "resources": resources,
            "cooperation_level": cooperation_level,
            "public_support": public_support,
            "risks": risks,
            "is_directly_affected": is_directly_affected,
            "importance": self._calculate_response_importance(
                country, event, response_type, intensity
            )
        }
        
        return response
    
    def _generate_organization_responses(self, event: Event,
                                        world_state: Dict[str, Any],
                                        political_state: Optional[Dict[str, Any]] = None,
                                        economic_state: Optional[Dict[str, Any]] = None,
                                        rng: Optional[random.Random] = None) -> List[Dict[str, Any]]:
        """
        生成国际组织响应
        
        Args:
            event: 事件对象
            world_state: 世界状态
            political_state: 政治状态
            economic_state: 经济状态
            rng: 随机数生成器
            
        Returns:
            国际组织响应列表
        """
        if rng is None:
            rng = random.Random()
        
        responses = []
        
        # 根据事件类型选择相关的国际组织
        relevant_organizations = self._get_relevant_organizations(event)
        
        for org in relevant_organizations:
            # 决定是否响应
            response_probability = self._calculate_organization_response_probability(
                org, event, political_state
            )
            
            if rng.random() < response_probability:
                response = self._generate_single_organization_response(
                    org, event, world_state, political_state, economic_state, rng
                )
                if response:
                    responses.append(response)
        
        # 限制每个事件的国际组织响应数量
        max_org_responses = self.config.get("max_organization_responses_per_event", 5)
        if len(responses) > max_org_responses:
            # 按响应重要性排序
            responses.sort(key=lambda r: r.get("importance", 0), reverse=True)
            responses = responses[:max_org_responses]
        
        return responses
    
    def _generate_single_organization_response(self, organization: str,
                                             event: Event,
                                             world_state: Dict[str, Any],
                                             political_state: Optional[Dict[str, Any]] = None,
                                             economic_state: Optional[Dict[str, Any]] = None,
                                             rng: Optional[random.Random] = None) -> Optional[Dict[str, Any]]:
        """
        生成单个国际组织的响应
        
        Args:
            organization: 组织名称
            event: 事件对象
            world_state: 世界状态
            political_state: 政治状态
            economic_state: 经济状态
            rng: 随机数生成器
            
        Returns:
            国际组织响应字典
        """
        if rng is None:
            rng = random.Random()
        
        # 获取国际组织能力
        org_capabilities = self.country_capabilities.get("International_Organizations", {
            "coordination_capability": 0.7,
            "resource_mobilization": 0.6,
            "legitimacy": 0.75,
            "response_speed": 0.5,
            "crisis_experience": 0.8
        })
        
        # 特定组织能力调整
        org_specific_capabilities = self._get_organization_specific_capabilities(organization)
        for key, value in org_specific_capabilities.items():
            org_capabilities[key] = value
        
        # 确定响应类型
        response_types = self._get_appropriate_organization_response_types(event, organization)
        
        if not response_types:
            return None
        
        # 选择响应类型
        response_type = rng.choice(response_types)
        
        # 计算响应强度
        intensity = self._calculate_organization_response_intensity(
            organization, event, org_capabilities
        )
        
        # 调整响应强度
        intensity = min(max(intensity + rng.uniform(-0.1, 0.1), 0.1), 1.0)
        
        # 计算响应及时性
        timeliness = org_capabilities.get("response_speed", 0.5) * 0.8 + rng.uniform(0.0, 0.2)
        
        # 计算响应有效性
        effectiveness = self._calculate_organization_response_effectiveness(
            organization, event, response_type, intensity, org_capabilities, rng
        )
        
        # 确定成员国支持度
        member_support = self._calculate_member_support(
            organization, event, political_state, rng
        )
        
        # 计算资源动员
        resources = self._calculate_organization_resource_mobilization(
            organization, event, intensity, org_capabilities, economic_state
        )
        
        # 生成响应描述
        response_description = self._generate_organization_response_description(
            organization, event, response_type
        )
        
        # 评估协调效率
        coordination_efficiency = org_capabilities.get("coordination_capability", 0.7) * \
                                 member_support * 0.8 + rng.uniform(0.0, 0.2)
        
        response = {
            "actor_type": "organization",
            "actor": organization,
            "event_id": event.id,
            "response_type": response_type,
            "description": response_description,
            "intensity": intensity,
            "timeliness": timeliness,
            "effectiveness": effectiveness,
            "resources": resources,
            "member_support": member_support,
            "coordination_efficiency": coordination_efficiency,
            "legitimacy": org_capabilities.get("legitimacy", 0.75),
            "importance": self._calculate_organization_response_importance(
                organization, event, response_type
            )
        }
        
        return response
    
    def _identify_affected_countries(self, event: Event,
                                   world_state: Dict[str, Any]) -> List[str]:
        """
        识别受事件直接影响的国家
        
        Args:
            event: 事件对象
            world_state: 世界状态
            
        Returns:
            受影响国家列表
        """
        affected_countries = []
        
        # 根据事件类型和地区确定受影响国家
        if event.region in ["North_America", "United_States"]:
            affected_countries.extend(["United_States", "Canada"])
        elif event.region in ["Europe", "European_Union"]:
            affected_countries.append("European_Union")
        elif event.region == "Asia":
            affected_countries.extend(["China", "Japan", "India"])
        elif event.region == "Middle_East":
            # 中东事件可能影响全球能源市场
            pass  # 不直接添加国家，让国际响应处理
        
        # 特定事件类型的处理
        if event.type == EventType.CONFLICT and "participants" in event.metadata:
            # 添加冲突参与国
            for country in event.metadata["participants"]:
                # 标准化国家名称
                if country == "US":
                    country = "United_States"
                elif country in ["UK", "Britain"]:
                    country = "European_Union"  # 简化处理，假设英国仍在欧盟框架内
                
                if country in self.country_capabilities:
                    affected_countries.append(country)
        
        # 全球性事件影响所有主要国家
        if event.scope == "global" or event.severity in [EventSeverity.MAJOR, EventSeverity.EXTREME]:
            major_countries = [c for c in self.country_capabilities.keys() 
                             if c != "International_Organizations"]
            affected_countries = list(set(affected_countries + major_countries))
        
        return affected_countries
    
    def _calculate_response_probability(self, country: str,
                                      event: Event,
                                      world_state: Dict[str, Any],
                                      political_state: Optional[Dict[str, Any]] = None) -> float:
        """
        计算国家响应概率
        
        Args:
            country: 国家名称
            event: 事件对象
            world_state: 世界状态
            political_state: 政治状态
            
        Returns:
            响应概率
        """
        # 基础概率
        base_probability = 0.3
        
        # 获取国家能力
        capabilities = self.country_capabilities.get(country, {})
        
        # 基于危机严重程度调整
        severity_factor = {
            EventSeverity.MINOR: 0.1,
            EventSeverity.MODERATE: 0.3,
            EventSeverity.MAJOR: 0.6,
            EventSeverity.EXTREME: 0.9
        }[event.severity]
        
        # 基于国家能力调整
        capability_factor = capabilities.get("crisis_experience", 0.5) * 0.3 + \
                          capabilities.get("cooperation_willingness", 0.5) * 0.2
        
        # 基于事件类型调整
        type_factor = self._get_event_type_response_factor(event.type)
        
        # 计算最终概率
        probability = base_probability + severity_factor * 0.5 + capability_factor * 0.2 + type_factor
        
        # 确保概率在合理范围内
        probability = min(max(probability, 0.05), 0.95)
        
        return probability
    
    def _get_event_type_response_factor(self, event_type: EventType) -> float:
        """
        获取事件类型的响应因子
        
        Args:
            event_type: 事件类型
            
        Returns:
            响应因子
        """
        factors = {
            EventType.ECONOMIC_CRISIS: 0.1,
            EventType.TECHNOLOGICAL_BREAKTHROUGH: -0.1,  # 技术突破通常不需要紧急响应
            EventType.CONFLICT: 0.2,
            EventType.PANDEMIC: 0.25,
            EventType.NATURAL_DISASTER: 0.2,
            EventType.POLITICAL_CRISIS: 0.1,
            EventType.RESOURCE_CRISIS: 0.15,
            EventType.SOCIAL_UNREST: 0.05
        }
        
        return factors.get(event_type, 0.1)
    
    def _get_appropriate_response_types(self, event: Event) -> List[ResponseType]:
        """
        获取适合事件的响应类型列表
        
        Args:
            event: 事件对象
            
        Returns:
            响应类型列表
        """
        response_mapping = {
            EventType.ECONOMIC_CRISIS: [
                ResponseType.ECONOMIC_STIMULUS,
                ResponseType.REGULATORY_ACTION,
                ResponseType.COOPERATION,
                ResponseType.DIPLOMATIC_EFFORT
            ],
            EventType.TECHNOLOGICAL_BREAKTHROUGH: [
                ResponseType.INVESTMENT,
                ResponseType.REGULATORY_ACTION,
                ResponseType.KNOWLEDGE_SHARING,
                ResponseType.COOPERATION
            ],
            EventType.CONFLICT: [
                ResponseType.DIPLOMATIC_EFFORT,
                ResponseType.ECONOMIC_SANCTIONS,
                ResponseType.MILITARY_INTERVENTION,
                ResponseType.HUMANITARIAN_AID,
                ResponseType.PEACEKEEPING
            ],
            EventType.PANDEMIC: [
                ResponseType.PUBLIC_HEALTH_MEASURES,
                ResponseType.HUMANITARIAN_AID,
                ResponseType.RESEARCH_FUNDING,
                ResponseType.COOPERATION
            ],
            EventType.NATURAL_DISASTER: [
                ResponseType.HUMANITARIAN_AID,
                ResponseType.RESCUE_OPERATION,
                ResponseType.RECONSTRUCTION,
                ResponseType.COOPERATION
            ],
            EventType.POLITICAL_CRISIS: [
                ResponseType.DIPLOMATIC_EFFORT,
                ResponseType.SANCTIONS,
                ResponseType.HUMANITARIAN_AID,
                ResponseType.MONITORING
            ],
            EventType.RESOURCE_CRISIS: [
                ResponseType.RESOURCE_ALLOCATION,
                ResponseType.ECONOMIC_MEASURES,
                ResponseType.TECHNOLOGICAL_ADAPTATION,
                ResponseType.COOPERATION
            ],
            EventType.SOCIAL_UNREST: [
                ResponseType.POLICING_ACTION,
                ResponseType.SOCIAL_REFORMS,
                ResponseType.DIPLOMATIC_EFFORT,
                ResponseType.MONITORING
            ]
        }
        
        return response_mapping.get(event.type, [ResponseType.MONITORING])
    
    def _get_appropriate_organization_response_types(self, event: Event,
                                                   organization: str) -> List[ResponseType]:
        """
        获取适合特定组织和事件的响应类型列表
        
        Args:
            event: 事件对象
            organization: 组织名称
            
        Returns:
            响应类型列表
        """
        # 基础响应类型
        base_types = self._get_appropriate_response_types(event)
        
        # 根据组织类型过滤
        org_specific_mapping = {
            "UN": [
                ResponseType.DIPLOMATIC_EFFORT,
                ResponseType.PEACEKEEPING,
                ResponseType.HUMANITARIAN_AID,
                ResponseType.MONITORING
            ],
            "WHO": [
                ResponseType.PUBLIC_HEALTH_MEASURES,
                ResponseType.RESEARCH_FUNDING,
                ResponseType.KNOWLEDGE_SHARING,
                ResponseType.COOPERATION
            ],
            "IMF": [
                ResponseType.ECONOMIC_STIMULUS,
                ResponseType.ECONOMIC_MEASURES,
                ResponseType.REGULATORY_ACTION
            ],
            "World_Bank": [
                ResponseType.INVESTMENT,
                ResponseType.RECONSTRUCTION,
                ResponseType.ECONOMIC_MEASURES
            ],
            "WTO": [
                ResponseType.REGULATORY_ACTION,
                ResponseType.TRADE_AGREEMENTS,
                ResponseType.COOPERATION
            ],
            "NATO": [
                ResponseType.MILITARY_INTERVENTION,
                ResponseType.PEACEKEEPING,
                ResponseType.MONITORING
            ],
            "EU": [
                ResponseType.ECONOMIC_STIMULUS,
                ResponseType.COOPERATION,
                ResponseType.REGULATORY_ACTION,
                ResponseType.HUMANITARIAN_AID
            ],
            "ASEAN": [
                ResponseType.DIPLOMATIC_EFFORT,
                ResponseType.COOPERATION,
                ResponseType.HUMANITARIAN_AID
            ],
            "African_Union": [
                ResponseType.PEACEKEEPING,
                ResponseType.HUMANITARIAN_AID,
                ResponseType.DIPLOMATIC_EFFORT
            ],
            "G20": [
                ResponseType.ECONOMIC_MEASURES,
                ResponseType.COOPERATION,
                ResponseType.DIPLOMATIC_EFFORT
            ]
        }
        
        # 组织特定的响应类型
        org_specific_types = org_specific_mapping.get(organization, [])
        
        # 找出两者的交集
        appropriate_types = [t for t in base_types if t in org_specific_types]
        
        # 如果没有交集，返回基础类型中的一些通用类型
        if not appropriate_types:
            appropriate_types = [t for t in base_types 
                               if t in [ResponseType.COOPERATION, ResponseType.MONITORING]]
        
        # 确保至少有一个响应类型
        if not appropriate_types and base_types:
            appropriate_types = base_types[:2]  # 返回基础类型的前两个
        
        return appropriate_types
    
    def _calculate_response_intensity(self, country: str,
                                    event: Event,
                                    is_directly_affected: bool,
                                    capabilities: Dict[str, float]) -> float:
        """
        计算国家响应强度
        
        Args:
            country: 国家名称
            event: 事件对象
            is_directly_affected: 是否直接受影响
            capabilities: 国家能力
            
        Returns:
            响应强度（0-1）
        """
        # 基础强度
        base_intensity = 0.5
        
        # 直接受影响时强度增加
        if is_directly_affected:
            base_intensity += 0.3
        
        # 基于事件严重程度调整
        severity_factor = {
            EventSeverity.MINOR: 0.1,
            EventSeverity.MODERATE: 0.3,
            EventSeverity.MAJOR: 0.6,
            EventSeverity.EXTREME: 0.9
        }[event.severity]
        base_intensity += severity_factor * 0.2
        
        # 基于国家能力调整
        capability_factor = (
            capabilities.get("economic_power", 0.5) * 0.3 +
            capabilities.get("military_power", 0.5) * 0.2 +
            capabilities.get("soft_power", 0.5) * 0.2 +
            capabilities.get("crisis_experience", 0.5) * 0.3
        )
        base_intensity += (capability_factor - 0.5) * 0.2  # 能力偏差调整
        
        # 确保强度在合理范围内
        intensity = min(max(base_intensity, 0.1), 1.0)
        
        return intensity
    
    def _calculate_organization_response_intensity(self, organization: str,
                                                 event: Event,
                                                 capabilities: Dict[str, float]) -> float:
        """
        计算国际组织响应强度
        
        Args:
            organization: 组织名称
            event: 事件对象
            capabilities: 组织能力
            
        Returns:
            响应强度（0-1）
        """
        # 基础强度
        base_intensity = 0.6
        
        # 基于事件严重程度调整
        severity_factor = {
            EventSeverity.MINOR: 0.2,
            EventSeverity.MODERATE: 0.4,
            EventSeverity.MAJOR: 0.7,
            EventSeverity.EXTREME: 0.9
        }[event.severity]
        base_intensity += (severity_factor - 0.5) * 0.3
        
        # 基于组织能力调整
        capability_factor = (
            capabilities.get("coordination_capability", 0.5) * 0.4 +
            capabilities.get("resource_mobilization", 0.5) * 0.3 +
            capabilities.get("crisis_experience", 0.5) * 0.3
        )
        base_intensity += (capability_factor - 0.5) * 0.2
        
        # 确保强度在合理范围内
        intensity = min(max(base_intensity, 0.2), 0.9)
        
        return intensity
    
    def _calculate_response_timeliness(self, country: str,
                                     event: Event,
                                     capabilities: Dict[str, float]) -> float:
        """
        计算响应及时性
        
        Args:
            country: 国家名称
            event: 事件对象
            capabilities: 国家能力
            
        Returns:
            及时性（0-1）
        """
        # 基础及时性
        base_timeliness = 0.6
        
        # 危机经验影响响应速度
        crisis_experience = capabilities.get("crisis_experience", 0.5)
        base_timeliness += (crisis_experience - 0.5) * 0.3
        
        # 危机严重程度越高，响应越快
        severity_factor = {
            EventSeverity.MINOR: 0.1,
            EventSeverity.MODERATE: 0.3,
            EventSeverity.MAJOR: 0.6,
            EventSeverity.EXTREME: 0.8
        }[event.severity]
        base_timeliness += (severity_factor - 0.5) * 0.2
        
        # 确保及时性在合理范围内
        timeliness = min(max(base_timeliness, 0.1), 0.95)
        
        return timeliness
    
    def _calculate_response_effectiveness(self, country: str,
                                        event: Event,
                                        response_type: ResponseType,
                                        intensity: float,
                                        capabilities: Dict[str, float],
                                        rng: random.Random) -> ResponseEffectiveness:
        """
        计算响应有效性
        
        Args:
            country: 国家名称
            event: 事件对象
            response_type: 响应类型
            intensity: 响应强度
            capabilities: 国家能力
            rng: 随机数生成器
            
        Returns:
            响应有效性枚举
        """
        # 计算基础有效性分数
        base_score = intensity * 0.5  # 强度占50%
        
        # 基于国家能力调整
        if response_type in [ResponseType.ECONOMIC_STIMULUS, ResponseType.ECONOMIC_MEASURES]:
            base_score += capabilities.get("economic_power", 0.5) * 0.3
        elif response_type in [ResponseType.MILITARY_INTERVENTION, ResponseType.PEACEKEEPING]:
            base_score += capabilities.get("military_power", 0.5) * 0.3
        elif response_type in [ResponseType.DIPLOMATIC_EFFORT, ResponseType.SANCTIONS]:
            base_score += capabilities.get("soft_power", 0.5) * 0.3
        elif response_type in [ResponseType.HUMANITARIAN_AID, ResponseType.RESCUE_OPERATION]:
            base_score += capabilities.get("resilience", 0.5) * 0.2 + capabilities.get("cooperation_willingness", 0.5) * 0.1
        
        # 危机经验加成
        base_score += capabilities.get("crisis_experience", 0.5) * 0.2
        
        # 添加一些随机性
        random_factor = rng.uniform(-0.1, 0.1)
        final_score = base_score + random_factor
        
        # 映射到有效性枚举
        if final_score >= 0.8:
            return ResponseEffectiveness.HIGH
        elif final_score >= 0.6:
            return ResponseEffectiveness.MEDIUM_HIGH
        elif final_score >= 0.4:
            return ResponseEffectiveness.MEDIUM
        elif final_score >= 0.2:
            return ResponseEffectiveness.MEDIUM_LOW
        else:
            return ResponseEffectiveness.LOW
    
    def _calculate_organization_response_effectiveness(self, organization: str,
                                                    event: Event,
                                                    response_type: ResponseType,
                                                    intensity: float,
                                                    capabilities: Dict[str, float],
                                                    rng: random.Random) -> ResponseEffectiveness:
        """
        计算国际组织响应有效性
        
        Args:
            organization: 组织名称
            event: 事件对象
            response_type: 响应类型
            intensity: 响应强度
            capabilities: 组织能力
            rng: 随机数生成器
            
        Returns:
            响应有效性枚举
        """
        # 计算基础有效性分数
        base_score = intensity * 0.5
        
        # 基于组织能力调整
        if response_type in [ResponseType.COOPERATION, ResponseType.DIPLOMATIC_EFFORT]:
            base_score += capabilities.get("coordination_capability", 0.5) * 0.3
        elif response_type in [ResponseType.HUMANITARIAN_AID, ResponseType.INVESTMENT]:
            base_score += capabilities.get("resource_mobilization", 0.5) * 0.3
        elif response_type in [ResponseType.PEACEKEEPING, ResponseType.SANCTIONS]:
            base_score += capabilities.get("legitimacy", 0.5) * 0.3
        
        # 危机经验加成
        base_score += capabilities.get("crisis_experience", 0.5) * 0.2
        
        # 添加一些随机性
        random_factor = rng.uniform(-0.1, 0.1)
        final_score = base_score + random_factor
        
        # 映射到有效性枚举
        if final_score >= 0.75:
            return ResponseEffectiveness.HIGH
        elif final_score >= 0.55:
            return ResponseEffectiveness.MEDIUM_HIGH
        elif final_score >= 0.35:
            return ResponseEffectiveness.MEDIUM
        elif final_score >= 0.2:
            return ResponseEffectiveness.MEDIUM_LOW
        else:
            return ResponseEffectiveness.LOW
    
    def _calculate_resource_allocation(self, country: str,
                                     event: Event,
                                     intensity: float,
                                     capabilities: Dict[str, float],
                                     economic_state: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        计算资源分配
        
        Args:
            country: 国家名称
            event: 事件对象
            intensity: 响应强度
            capabilities: 国家能力
            economic_state: 经济状态
            
        Returns:
            资源分配字典
        """
        # 基础资源比例（占GDP的百分比）
        base_economic_allocation = 0.01 * intensity
        
        # 基于经济实力调整
        economic_power = capabilities.get("economic_power", 0.5)
        economic_allocation = base_economic_allocation * (1 + (economic_power - 0.5) * 2)
        
        # 军事资源分配（仅适用于军事相关响应）
        military_allocation = 0.0
        if event.type in [EventType.CONFLICT] and intensity > 0.5:
            military_power = capabilities.get("military_power", 0.5)
            military_allocation = 0.005 * intensity * military_power
        
        # 人力资源分配
        human_resource_allocation = 0.02 * intensity  # 占总劳动力的百分比
        
        resources = {
            "economic": round(economic_allocation, 4),
            "military": round(military_allocation, 4),
            "human": round(human_resource_allocation, 4),
            "time": {
                "planning": 1.0 + (1 - capabilities.get("crisis_experience", 0.5)) * 2,  # 周
                "implementation": 4.0 * intensity,  # 周
                "sustainment": 52.0 * intensity  # 周（持续时间）
            }
        }
        
        return resources
    
    def _calculate_organization_resource_mobilization(self, organization: str,
                                                    event: Event,
                                                    intensity: float,
                                                    capabilities: Dict[str, float],
                                                    economic_state: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        计算国际组织资源动员
        
        Args:
            organization: 组织名称
            event: 事件对象
            intensity: 响应强度
            capabilities: 组织能力
            economic_state: 经济状态
            
        Returns:
            资源动员字典
        """
        # 资源动员能力
        resource_mobilization = capabilities.get("resource_mobilization", 0.6)
        
        # 基础资金动员（十亿美元）
        base_funding = intensity * 1.0
        actual_funding = base_funding * resource_mobilization * 2
        
        # 人员动员
        personnel = 1000 * intensity * resource_mobilization * 2
        
        # 时间线
        mobilization_time = 2.0 / (capabilities.get("response_speed", 0.5) + 0.1)  # 周
        implementation_time = 8.0 * intensity  # 周
        
        resources = {
            "funding_billion_usd": round(actual_funding, 2),
            "personnel": round(personnel),
            "equipment": intensity * resource_mobilization * 50,  # 设备单位
            "time": {
                "mobilization": mobilization_time,  # 周
                "implementation": implementation_time,  # 周
                "sustainment": 52.0 * intensity  # 周
            }
        }
        
        return resources
    
    def _calculate_public_support(self, country: str,
                                event: Event,
                                response_type: ResponseType,
                                intensity: float,
                                rng: random.Random) -> float:
        """
        计算国内支持度
        
        Args:
            country: 国家名称
            event: 事件对象
            response_type: 响应类型
            intensity: 响应强度
            rng: 随机数生成器
            
        Returns:
            支持度（0-1）
        """
        # 基础支持度
        base_support = 0.5
        
        # 不同响应类型的支持度差异
        response_support_factors = {
            ResponseType.HUMANITARIAN_AID: 0.2,
            ResponseType.PUBLIC_HEALTH_MEASURES: 0.15,
            ResponseType.ECONOMIC_STIMULUS: 0.1,
            ResponseType.SOCIAL_REFORMS: 0.05,
            ResponseType.MILITARY_INTERVENTION: -0.15,
            ResponseType.ECONOMIC_SANCTIONS: -0.1,
            ResponseType.POLICING_ACTION: -0.05
        }
        
        support_factor = response_support_factors.get(response_type, 0.0)
        base_support += support_factor
        
        # 危机严重程度影响支持度
        severity_factor = {
            EventSeverity.MINOR: -0.1,
            EventSeverity.MODERATE: 0.0,
            EventSeverity.MAJOR: 0.1,
            EventSeverity.EXTREME: 0.2
        }[event.severity]
        base_support += severity_factor
        
        # 高强度响应可能降低支持度（资源消耗）
        if intensity > 0.7:
            base_support -= (intensity - 0.7) * 0.3
        
        # 添加一些随机性
        base_support += rng.uniform(-0.1, 0.1)
        
        # 确保支持度在合理范围内
        support = min(max(base_support, 0.1), 0.9)
        
        return support
    
    def _calculate_member_support(self, organization: str,
                                event: Event,
                                political_state: Optional[Dict[str, Any]] = None,
                                rng: random.Random = None) -> float:
        """
        计算成员国支持度
        
        Args:
            organization: 组织名称
            event: 事件对象
            political_state: 政治状态
            rng: 随机数生成器
            
        Returns:
            支持度（0-1）
        """
        if rng is None:
            rng = random.Random()
        
        # 基础支持度
        base_support = 0.6
        
        # 不同组织的基础凝聚力不同
        org_cohesion = {
            "EU": 0.75,
            "NATO": 0.7,
            "ASEAN": 0.6,
            "African_Union": 0.55,
            "G20": 0.5,
            "UN": 0.65,
            "WHO": 0.7,
            "IMF": 0.65,
            "World_Bank": 0.65,
            "WTO": 0.6
        }
        
        cohesion = org_cohesion.get(organization, 0.6)
        base_support = cohesion
        
        # 危机严重程度影响支持度
        severity_factor = {
            EventSeverity.MINOR: -0.05,
            EventSeverity.MODERATE: 0.0,
            EventSeverity.MAJOR: 0.05,
            EventSeverity.EXTREME: 0.1
        }[event.severity]
        base_support += severity_factor
        
        # 添加一些随机性
        base_support += rng.uniform(-0.05, 0.05)
        
        # 确保支持度在合理范围内
        support = min(max(base_support, 0.3), 0.95)
        
        return support
    
    def _generate_response_description(self, country: str,
                                     event: Event,
                                     response_type: ResponseType,
                                     intensity: float) -> str:
        """
        生成响应描述
        
        Args:
            country: 国家名称
            event: 事件对象
            response_type: 响应类型
            intensity: 响应强度
            
        Returns:
            响应描述字符串
        """
        # 响应强度描述
        intensity_descriptions = {
            (0.0, 0.2): "象征性",
            (0.2, 0.4): "有限",
            (0.4, 0.6): "适度",
            (0.6, 0.8): "积极",
            (0.8, 1.1): "全面"
        }
        
        intensity_desc = ""
        for (min_val, max_val), desc in intensity_descriptions.items():
            if min_val <= intensity < max_val:
                intensity_desc = desc
                break
        
        # 响应类型描述
        response_descriptions = {
            ResponseType.ECONOMIC_STIMULUS: "经济刺激计划",
            ResponseType.ECONOMIC_SANCTIONS: "经济制裁",
            ResponseType.MILITARY_INTERVENTION: "军事干预",
            ResponseType.HUMANITARIAN_AID: "人道主义援助",
            ResponseType.DIPLOMATIC_EFFORT: "外交努力",
            ResponseType.PUBLIC_HEALTH_MEASURES: "公共卫生措施",
            ResponseType.REGULATORY_ACTION: "监管行动",
            ResponseType.INVESTMENT: "投资",
            ResponseType.SOCIAL_REFORMS: "社会改革",
            ResponseType.TECHNOLOGICAL_ADAPTATION: "技术适应",
            ResponseType.RESOURCE_ALLOCATION: "资源分配",
            ResponseType.KNOWLEDGE_SHARING: "知识共享",
            ResponseType.RESCUE_OPERATION: "救援行动",
            ResponseType.RECONSTRUCTION: "重建",
            ResponseType.PEACEKEEPING: "维和行动",
            ResponseType.POLICING_ACTION: "治安行动",
            ResponseType.SANCTIONS: "制裁",
            ResponseType.COOPERATION: "国际合作",
            ResponseType.MONITORING: "监测",
            ResponseType.TRADE_AGREEMENTS: "贸易协议",
            ResponseType.ECONOMIC_MEASURES: "经济措施",
            ResponseType.RESEARCH_FUNDING: "研究 funding"
        }
        
        response_desc = response_descriptions.get(response_type, "响应")
        
        # 事件描述前缀
        if event.type == EventType.NATURAL_DISASTER:
            event_prefix = f"应对{event.region}的{event.subtype}"
        elif event.type == EventType.CONFLICT and "participants" in event.metadata:
            participants = "与".join(event.metadata["participants"])
            event_prefix = f"应对{participants}之间的{event.subtype}"
        else:
            event_prefix = f"应对{event.description}"
        
        description = f"{country}宣布对{event_prefix}采取{intensity_desc}{response_desc}"
        
        return description
    
    def _generate_organization_response_description(self, organization: str,
                                                  event: Event,
                                                  response_type: ResponseType) -> str:
        """
        生成国际组织响应描述
        
        Args:
            organization: 组织名称
            event: 事件对象
            response_type: 响应类型
            
        Returns:
            响应描述字符串
        """
        # 响应类型描述（组织版本）
        response_descriptions = {
            ResponseType.ECONOMIC_STIMULUS: "经济援助计划",
            ResponseType.ECONOMIC_SANCTIONS: "经济制裁决议",
            ResponseType.MILITARY_INTERVENTION: "军事行动授权",
            ResponseType.HUMANITARIAN_AID: "人道主义援助行动",
            ResponseType.DIPLOMATIC_EFFORT: "外交调解",
            ResponseType.PUBLIC_HEALTH_MEASURES: "公共卫生响应",
            ResponseType.REGULATORY_ACTION: "监管框架",
            ResponseType.INVESTMENT: "发展基金",
            ResponseType.SOCIAL_REFORMS: "社会发展项目",
            ResponseType.TECHNOLOGICAL_ADAPTATION: "技术援助",
            ResponseType.RESOURCE_ALLOCATION: "资源协调",
            ResponseType.KNOWLEDGE_SHARING: "知识共享平台",
            ResponseType.RESCUE_OPERATION: "紧急救援协调",
            ResponseType.RECONSTRUCTION: "重建项目",
            ResponseType.PEACEKEEPING: "维和任务",
            ResponseType.POLICING_ACTION: "安全支持",
            ResponseType.SANCTIONS: "制裁决议",
            ResponseType.COOPERATION: "国际合作框架",
            ResponseType.MONITORING: "监测任务",
            ResponseType.TRADE_AGREEMENTS: "贸易协议",
            ResponseType.ECONOMIC_MEASURES: "经济政策协调",
            ResponseType.RESEARCH_FUNDING: "研究资助计划"
        }
        
        response_desc = response_descriptions.get(response_type, "响应")
        
        # 组织名称格式化
        org_name_formatted = self._format_organization_name(organization)
        
        description = f"{org_name_formatted}启动{response_desc}以应对{event.description}"
        
        return description
    
    def _format_organization_name(self, organization: str) -> str:
        """
        格式化组织名称
        
        Args:
            organization: 组织名称
            
        Returns:
            格式化后的组织名称
        """
        name_mapping = {
            "UN": "联合国",
            "WHO": "世界卫生组织",
            "IMF": "国际货币基金组织",
            "World_Bank": "世界银行",
            "WTO": "世界贸易组织",
            "NATO": "北大西洋公约组织",
            "EU": "欧盟",
            "ASEAN": "东南亚国家联盟",
            "African_Union": "非洲联盟",
            "G20": "二十国集团"
        }
        
        return name_mapping.get(organization, organization)
    
    def _assess_response_risks(self, country: str,
                             event: Event,
                             response_type: ResponseType,
                             intensity: float) -> Dict[str, float]:
        """
        评估响应风险
        
        Args:
            country: 国家名称
            event: 事件对象
            response_type: 响应类型
            intensity: 响应强度
            
        Returns:
            风险字典
        """
        risks = {
            "economic_risk": 0.0,
            "political_risk": 0.0,
            "reputation_risk": 0.0,
            "security_risk": 0.0
        }
        
        # 经济风险
        if response_type in [
            ResponseType.ECONOMIC_STIMULUS,
            ResponseType.HUMANITARIAN_AID,
            ResponseType.INVESTMENT,
            ResponseType.RECONSTRUCTION
        ]:
            risks["economic_risk"] = intensity * 0.6
        
        # 政治风险
        if response_type in [
            ResponseType.MILITARY_INTERVENTION,
            ResponseType.ECONOMIC_SANCTIONS,
            ResponseType.POLICING_ACTION,
            ResponseType.SANCTIONS
        ]:
            risks["political_risk"] = intensity * 0.7
        
        # 声誉风险
        if response_type in [
            ResponseType.MILITARY_INTERVENTION,
            ResponseType.ECONOMIC_SANCTIONS,
            ResponseType.SANCTIONS,
            ResponseType.DIPLOMATIC_EFFORT
        ]:
            risks["reputation_risk"] = intensity * 0.5
        
        # 安全风险
        if response_type in [
            ResponseType.MILITARY_INTERVENTION,
            ResponseType.POLICING_ACTION,
            ResponseType.RESCUE_OPERATION,
            ResponseType.PEACEKEEPING
        ]:
            risks["security_risk"] = intensity * 0.8
        
        # 冲突事件增加所有风险
        if event.type == EventType.CONFLICT:
            for key in risks:
                risks[key] = min(risks[key] * 1.5, 1.0)
        
        # 确保风险在合理范围内
        for key in risks:
            risks[key] = min(max(risks[key], 0.0), 1.0)
        
        return risks
    
    def _calculate_response_importance(self, country: str,
                                     event: Event,
                                     response_type: ResponseType,
                                     intensity: float) -> float:
        """
        计算响应重要性
        
        Args:
            country: 国家名称
            event: 事件对象
            response_type: 响应类型
            intensity: 响应强度
            
        Returns:
            重要性分数（0-1）
        """
        # 基础重要性
        importance = 0.5
        
        # 事件严重程度权重
        severity_weight = {
            EventSeverity.MINOR: 0.1,
            EventSeverity.MODERATE: 0.3,
            EventSeverity.MAJOR: 0.7,
            EventSeverity.EXTREME: 0.9
        }[event.severity]
        importance += severity_weight * 0.3
        
        # 国家影响力权重（P5+国家更重要）
        major_powers = ["United_States", "China", "Russia", "European_Union"]
        if country in major_powers:
            importance += 0.2
        
        # 响应强度权重
        importance += intensity * 0.1
        
        # 确保重要性在合理范围内
        importance = min(max(importance, 0.1), 1.0)
        
        return importance
    
    def _calculate_organization_response_importance(self, organization: str,
                                                  event: Event,
                                                  response_type: ResponseType) -> float:
        """
        计算国际组织响应重要性
        
        Args:
            organization: 组织名称
            event: 事件对象
            response_type: 响应类型
            
        Returns:
            重要性分数（0-1）
        """
        # 基础重要性
        importance = 0.6
        
        # 事件严重程度权重
        severity_weight = {
            EventSeverity.MINOR: 0.1,
            EventSeverity.MODERATE: 0.3,
            EventSeverity.MAJOR: 0.7,
            EventSeverity.EXTREME: 0.9
        }[event.severity]
        importance += severity_weight * 0.2
        
        # 组织重要性权重
        org_importance = {
            "UN": 0.3,
            "G20": 0.25,
            "IMF": 0.2,
            "World_Bank": 0.2,
            "WHO": 0.25,
            "EU": 0.2,
            "NATO": 0.2,
            "ASEAN": 0.1,
            "African_Union": 0.1,
            "WTO": 0.15
        }
        
        importance += org_importance.get(organization, 0.1)
        
        # 确保重要性在合理范围内
        importance = min(max(importance, 0.2), 1.0)
        
        return importance
    
    def _get_relevant_organizations(self, event: Event) -> List[str]:
        """
        获取与事件相关的国际组织
        
        Args:
            event: 事件对象
            
        Returns:
            相关组织列表
        """
        # 基础相关组织
        relevant = ["UN", "G20"]
        
        # 根据事件类型添加特定组织
        if event.type == EventType.ECONOMIC_CRISIS:
            relevant.extend(["IMF", "World_Bank", "WTO"])
        elif event.type == EventType.CONFLICT:
            relevant.extend(["NATO", "EU"])
        elif event.type == EventType.PANDEMIC:
            relevant.extend(["WHO"])
        elif event.type == EventType.NATURAL_DISASTER:
            relevant.extend(["UN", "World_Bank"])
        elif event.type == EventType.POLITICAL_CRISIS:
            relevant.extend(["UN", "EU"])
        elif event.type == EventType.RESOURCE_CRISIS:
            relevant.extend(["UN", "WTO"])
        elif event.type == EventType.SOCIAL_UNREST:
            relevant.extend(["UN"])
        elif event.type == EventType.TECHNOLOGICAL_BREAKTHROUGH:
            relevant.extend(["G20"])
        
        # 根据地区添加区域组织
        if event.region in ["Europe", "European_Union"]:
            relevant.append("EU")
        elif event.region == "Asia":
            relevant.append("ASEAN")
        elif event.region == "Africa":
            relevant.append("African_Union")
        
        # 去重并确保在定义的组织列表中
        relevant = list(set([org for org in relevant if org in self.international_organizations]))
        
        return relevant
    
    def _calculate_organization_response_probability(self, organization: str,
                                                   event: Event,
                                                   political_state: Optional[Dict[str, Any]] = None) -> float:
        """
        计算国际组织响应概率
        
        Args:
            organization: 组织名称
            event: 事件对象
            political_state: 政治状态
            
        Returns:
            响应概率
        """
        # 基础概率
        base_probability = 0.5
        
        # 事件严重程度影响
        severity_factor = {
            EventSeverity.MINOR: 0.1,
            EventSeverity.MODERATE: 0.3,
            EventSeverity.MAJOR: 0.6,
            EventSeverity.EXTREME: 0.9
        }[event.severity]
        base_probability += (severity_factor - 0.5) * 0.3
        
        # 组织相关性影响
        relevant_orgs = self._get_relevant_organizations(event)
        if organization in relevant_orgs:
            base_probability += 0.2
        
        # 确保概率在合理范围内
        probability = min(max(base_probability, 0.1), 0.9)
        
        return probability
    
    def _get_organization_specific_capabilities(self, organization: str) -> Dict[str, float]:
        """
        获取特定组织的能力
        
        Args:
            organization: 组织名称
            
        Returns:
            能力字典
        """
        org_capabilities = {
            "UN": {
                "coordination_capability": 0.75,
                "resource_mobilization": 0.6,
                "legitimacy": 0.8,
                "response_speed": 0.45,
                "crisis_experience": 0.85
            },
            "WHO": {
                "coordination_capability": 0.8,
                "resource_mobilization": 0.65,
                "legitimacy": 0.85,
                "response_speed": 0.6,
                "crisis_experience": 0.9
            },
            "IMF": {
                "coordination_capability": 0.7,
                "resource_mobilization": 0.85,
                "legitimacy": 0.75,
                "response_speed": 0.65,
                "crisis_experience": 0.8
            },
            "World_Bank": {
                "coordination_capability": 0.65,
                "resource_mobilization": 0.8,
                "legitimacy": 0.7,
                "response_speed": 0.5,
                "crisis_experience": 0.75
            },
            "WTO": {
                "coordination_capability": 0.6,
                "resource_mobilization": 0.55,
                "legitimacy": 0.7,
                "response_speed": 0.4,
                "crisis_experience": 0.65
            },
            "NATO": {
                "coordination_capability": 0.75,
                "resource_mobilization": 0.7,
                "legitimacy": 0.65,
                "response_speed": 0.6,
                "crisis_experience": 0.8
            },
            "EU": {
                "coordination_capability": 0.8,
                "resource_mobilization": 0.75,
                "legitimacy": 0.75,
                "response_speed": 0.55,
                "crisis_experience": 0.7
            },
            "ASEAN": {
                "coordination_capability": 0.6,
                "resource_mobilization": 0.5,
                "legitimacy": 0.6,
                "response_speed": 0.45,
                "crisis_experience": 0.6
            },
            "African_Union": {
                "coordination_capability": 0.55,
                "resource_mobilization": 0.45,
                "legitimacy": 0.65,
                "response_speed": 0.4,
                "crisis_experience": 0.55
            },
            "G20": {
                "coordination_capability": 0.65,
                "resource_mobilization": 0.85,
                "legitimacy": 0.7,
                "response_speed": 0.5,
                "crisis_experience": 0.7
            }
        }
        
        return org_capabilities.get(organization, {})
    
    def _generate_economic_crisis_responses(self, event: Event,
                                          world_state: Dict[str, Any],
                                          political_state: Dict[str, Any],
                                          economic_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成经济危机响应
        
        Args:
            event: 事件对象
            world_state: 世界状态
            political_state: 政治状态
            economic_state: 经济状态
            
        Returns:
            特定响应策略列表
        """
        # 经济危机特定响应
        specific_responses = [
            {
                "type": "fiscal_stimulus",
                "description": "实施大规模财政刺激计划",
                "effect": "短期内提升GDP, 长期可能增加债务",
                "applicable_countries": ["United_States", "China", "European_Union", "Japan"]
            },
            {
                "type": "monetary_policy",
                "description": "调整货币政策（降息、量化宽松）",
                "effect": "增加市场流动性, 可能导致通胀",
                "applicable_countries": ["United_States", "European_Union", "Japan"]
            },
            {
                "type": "bank_bailout",
                "description": "为陷入困境的金融机构提供援助",
                "effect": "稳定金融系统, 增加政府债务",
                "applicable_countries": ["United_States", "European_Union", "Japan"]
            },
            {
                "type": "structural_reform",
                "description": "推行经济结构性改革",
                "effect": "长期提升经济效率, 短期可能有阵痛",
                "applicable_countries": ["China", "European_Union", "India"]
            },
            {
                "type": "trade_support",
                "description": "支持出口和国际贸易",
                "effect": "促进经济复苏, 可能引发贸易摩擦",
                "applicable_countries": ["China", "Japan", "Germany", "South_Korea"]
            }
        ]
        
        return specific_responses
    
    def _generate_technological_breakthrough_responses(self, event: Event,
                                                     world_state: Dict[str, Any],
                                                     political_state: Dict[str, Any],
                                                     economic_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成技术突破响应
        
        Args:
            event: 事件对象
            world_state: 世界状态
            political_state: 政治状态
            economic_state: 经济状态
            
        Returns:
            特定响应策略列表
        """
        # 技术突破特定响应
        specific_responses = [
            {
                "type": "research_funding",
                "description": "增加相关领域研究资金",
                "effect": "加速技术发展, 推动创新",
                "applicable_countries": ["United_States", "China", "European_Union", "Japan"]
            },
            {
                "type": "regulatory_framework",
                "description": "建立新技术监管框架",
                "effect": "平衡创新与安全, 可能限制发展速度",
                "applicable_countries": ["United_States", "European_Union", "China"]
            },
            {
                "type": "workforce_adaptation",
                "description": "促进劳动力适应新技术",
                "effect": "减少技术转型阻力, 提高生产力",
                "applicable_countries": ["All"]
            },
            {
                "type": "export_control",
                "description": "实施技术出口管制",
                "effect": "保护技术优势, 可能引发技术脱钩",
                "applicable_countries": ["United_States", "China", "European_Union"]
            },
            {
                "type": "international_collaboration",
                "description": "促进国际技术合作",
                "effect": "加速全球技术进步, 分享发展成果",
                "applicable_countries": ["European_Union", "Japan", "Canada", "Australia"]
            }
        ]
        
        return specific_responses
    
    def _generate_conflict_responses(self, event: Event,
                                   world_state: Dict[str, Any],
                                   political_state: Dict[str, Any],
                                   economic_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成冲突响应
        
        Args:
            event: 事件对象
            world_state: 世界状态
            political_state: 政治状态
            economic_state: 经济状态
            
        Returns:
            特定响应策略列表
        """
        # 冲突特定响应
        specific_responses = [
            {
                "type": "peace_mediation",
                "description": "进行和平调解与谈判",
                "effect": "寻求政治解决方案, 减少武装冲突",
                "applicable_countries": ["United_States", "European_Union", "UN", "Russia"]
            },
            {
                "type": "sanctions",
                "description": "对冲突方实施经济制裁",
                "effect": "增加冲突成本, 迫使妥协",
                "applicable_countries": ["United_States", "European_Union"]
            },
            {
                "type": "military_support",
                "description": "向冲突一方提供军事支持",
                "effect": "改变战场态势, 可能延长冲突",
                "applicable_countries": ["United_States", "Russia", "China"]
            },
            {
                "type": "peacekeeping",
                "description": "部署维和部队",
                "effect": "建立缓冲区, 保护平民",
                "applicable_countries": ["UN", "NATO", "European_Union", "African_Union"]
            },
            {
                "type": "humanitarian_corridor",
                "description": "建立人道主义走廊",
                "effect": "允许救援物资进入, 平民撤离",
                "applicable_countries": ["UN", "Red_Cross", "All"]
            }
        ]
        
        return specific_responses
    
    def _generate_pandemic_responses(self, event: Event,
                                   world_state: Dict[str, Any],
                                   political_state: Dict[str, Any],
                                   economic_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成疫情响应
        
        Args:
            event: 事件对象
            world_state: 世界状态
            political_state: 政治状态
            economic_state: 经济状态
            
        Returns:
            特定响应策略列表
        """
        # 疫情特定响应
        specific_responses = [
            {
                "type": "vaccine_development",
                "description": "加速疫苗研发和生产",
                "effect": "提供长期防控手段, 需要时间",
                "applicable_countries": ["United_States", "European_Union", "China", "Japan"]
            },
            {
                "type": "lockdown_measures",
                "description": "实施封锁和社交距离措施",
                "effect": "快速控制传播, 严重影响经济",
                "applicable_countries": ["All"]
            },
            {
                "type": "healthcare_expansion",
                "description": "扩大医疗系统容量",
                "effect": "提高治疗能力, 减少死亡率",
                "applicable_countries": ["All"]
            },
            {
                "type": "information_campaign",
                "description": "开展公共卫生信息宣传",
                "effect": "提高公众意识, 促进合作",
                "applicable_countries": ["All"]
            },
            {
                "type": "international_coordination",
                "description": "协调国际疫情应对",
                "effect": "共享资源和信息, 防止跨境传播",
                "applicable_countries": ["WHO", "G20", "All"]
            }
        ]
        
        return specific_responses
    
    def _generate_natural_disaster_responses(self, event: Event,
                                           world_state: Dict[str, Any],
                                           political_state: Dict[str, Any],
                                           economic_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成自然灾害响应
        
        Args:
            event: 事件对象
            world_state: 世界状态
            political_state: 政治状态
            economic_state: 经济状态
            
        Returns:
            特定响应策略列表
        """
        # 自然灾害特定响应
        specific_responses = [
            {
                "type": "emergency_relief",
                "description": "提供紧急救援物资和人员",
                "effect": "减轻即时痛苦, 拯救生命",
                "applicable_countries": ["All"]
            },
            {
                "type": "infrastructure_repair",
                "description": "修复受损基础设施",
                "effect": "恢复基本服务, 支持重建",
                "applicable_countries": ["All"]
            },
            {
                "type": "climate_adaptation",
                "description": "加强气候适应措施",
                "effect": "减少未来灾害影响, 需要长期投入",
                "applicable_countries": ["All"]
            },
            {
                "type": "early_warning",
                "description": "改进早期预警系统",
                "effect": "提前准备, 减少损失",
                "applicable_countries": ["All"]
            },
            {
                "type": "disaster_risk_reduction",
                "description": "实施灾害风险管理策略",
                "effect": "提高韧性, 降低脆弱性",
                "applicable_countries": ["All"]
            }
        ]
        
        return specific_responses
    
    def _generate_political_crisis_responses(self, event: Event,
                                          world_state: Dict[str, Any],
                                          political_state: Dict[str, Any],
                                          economic_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成政治危机响应
        
        Args:
            event: 事件对象
            world_state: 世界状态
            political_state: 政治状态
            economic_state: 经济状态
            
        Returns:
            特定响应策略列表
        """
        # 政治危机特定响应
        specific_responses = [
            {
                "type": "diplomatic_recognition",
                "description": "承认新政权或合法性",
                "effect": "影响冲突各方力量平衡",
                "applicable_countries": ["United_States", "European_Union", "China", "UN"]
            },
            {
                "type": "election_support",
                "description": "支持举行自由公平选举",
                "effect": "促进政治过渡, 建立合法性",
                "applicable_countries": ["UN", "European_Union", "United_States"]
            },
            {
                "type": "political_mediation",
                "description": "进行政治调解",
                "effect": "促进对话, 寻求妥协",
                "applicable_countries": ["UN", "European_Union", "African_Union", "ASEAN"]
            },
            {
                "type": "human_rights_monitoring",
                "description": "监测人权状况",
                "effect": "提高透明度, 施加国际压力",
                "applicable_countries": ["UN", "European_Union", "United_States"]
            },
            {
                "type": "sanctions",
                "description": "实施针对性制裁",
                "effect": "对违规行为施加成本",
                "applicable_countries": ["United_States", "European_Union", "UN"]
            }
        ]
        
        return specific_responses
    
    def _generate_resource_crisis_responses(self, event: Event,
                                          world_state: Dict[str, Any],
                                          political_state: Dict[str, Any],
                                          economic_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成资源危机响应
        
        Args:
            event: 事件对象
            world_state: 世界状态
            political_state: 政治状态
            economic_state: 经济状态
            
        Returns:
            特定响应策略列表
        """
        # 资源危机特定响应
        specific_responses = [
            {
                "type": "diversification",
                "description": "促进资源供应多元化",
                "effect": "减少对单一来源依赖",
                "applicable_countries": ["All"]
            },
            {
                "type": "conservation",
                "description": "实施资源节约措施",
                "effect": "降低需求, 延长资源使用",
                "applicable_countries": ["All"]
            },
            {
                "type": "renewable_investment",
                "description": "投资可再生资源",
                "effect": "长期解决资源短缺, 需要大规模投资",
                "applicable_countries": ["United_States", "European_Union", "China", "Japan"]
            },
            {
                "type": "strategic_reserves",
                "description": "建立战略资源储备",
                "effect": "应对短期供应中断",
                "applicable_countries": ["United_States", "China", "Japan", "European_Union"]
            },
            {
                "type": "international_cooperation",
                "description": "促进资源合作与公平分配",
                "effect": "减少冲突, 确保稳定供应",
                "applicable_countries": ["UN", "G20", "WTO"]
            }
        ]
        
        return specific_responses
    
    def _generate_social_unrest_responses(self, event: Event,
                                        world_state: Dict[str, Any],
                                        political_state: Dict[str, Any],
                                        economic_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成社会动荡响应
        
        Args:
            event: 事件对象
            world_state: 世界状态
            political_state: 政治状态
            economic_state: 经济状态
            
        Returns:
            特定响应策略列表
        """
        # 社会动荡特定响应
        specific_responses = [
            {
                "type": "dialogue_initiation",
                "description": "启动政府与抗议者对话",
                "effect": "寻求和平解决方案, 缓解紧张局势",
                "applicable_countries": ["All"]
            },
            {
                "type": "law_enforcement",
                "description": "动用执法力量维持秩序",
                "effect": "迅速恢复秩序, 可能激化矛盾",
                "applicable_countries": ["All"]
            },
            {
                "type": "economic_reforms",
                "description": "实施经济改革解决根本问题",
                "effect": "长期改善社会条件, 需要时间见效",
                "applicable_countries": ["All"]
            },
            {
                "type": "humanitarian_assistance",
                "description": "向受影响社区提供援助",
                "effect": "减轻民众痛苦, 改善政府形象",
                "applicable_countries": ["All"]
            },
            {
                "type": "international_observation",
                "description": "邀请国际观察员",
                "effect": "增加透明度, 施加国际标准",
                "applicable_countries": ["UN", "European_Union", "African_Union"]
            }
        ]
        
        return specific_responses
    
    def update_relations_after_response(self, response: Dict[str, Any],
                                      other_countries: List[str]) -> Dict[Tuple[str, str], float]:
        """
        更新国家关系矩阵
        
        Args:
            response: 响应字典
            other_countries: 其他国家列表
            
        Returns:
            更新后的关系矩阵增量
        """
        relation_changes = {}
        actor = response.get("actor")
        
        if actor in self.country_capabilities and actor != "International_Organizations":
            for other in other_countries:
                if other in self.country_capabilities and other != "International_Organizations" and other != actor:
                    # 计算关系变化
                    response_type = response.get("response_type")
                    effectiveness = response.get("effectiveness")
                    
                    change = 0.0
                    
                    # 不同响应类型对关系的影响
                    if response_type in [ResponseType.HUMANITARIAN_AID, ResponseType.COOPERATION]:
                        change = 0.05 * (effectiveness.value + 1) / 5  # 将枚举转换为-2到2的值
                    elif response_type in [ResponseType.ECONOMIC_SANCTIONS, ResponseType.SANCTIONS]:
                        if other == response.get("target", other):
                            change = -0.1 * (effectiveness.value + 1) / 5
                    elif response_type in [ResponseType.MILITARY_INTERVENTION]:
                        # 根据是否支持干预而变化
                        if self.country_relations.get((actor, other), 0) > 0:
                            change = 0.03
                        else:
                            change = -0.08
                    
                    # 保存关系变化
                    relation_changes[(actor, other)] = change
                    relation_changes[(other, actor)] = change
        
        return relation_changes
    
    def get_response_statistics(self, event_type: Optional[EventType] = None,
                              time_period: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        获取响应统计信息
        
        Args:
            event_type: 事件类型过滤
            time_period: 时间周期过滤
            
        Returns:
            统计信息字典
        """
        # 过滤响应历史
        filtered_responses = []
        for record in self.response_history:
            include = True
            
            if event_type and "event_type" in record.get("response", {}) and record["response"]["event_type"] != event_type:
                include = False
            
            # 时间过滤逻辑可以在这里添加
            
            if include:
                filtered_responses.append(record)
        
        # 计算统计信息
        total_responses = len(filtered_responses)
        
        # 按类型统计
        by_type = {}
        by_actor = {}
        effectiveness_count = {}
        
        for record in filtered_responses:
            response = record.get("response", {})
            
            # 按类型统计
            response_type = response.get("response_type")
            if response_type:
                by_type[response_type] = by_type.get(response_type, 0) + 1
            
            # 按行为者统计
            actor = response.get("actor")
            if actor:
                by_actor[actor] = by_actor.get(actor, 0) + 1
            
            # 按有效性统计
            effectiveness = response.get("effectiveness")
            if effectiveness:
                effectiveness_count[effectiveness] = effectiveness_count.get(effectiveness, 0) + 1
        
        stats = {
            "total_responses": total_responses,
            "responses_by_type": by_type,
            "responses_by_actor": by_actor,
            "effectiveness_distribution": effectiveness_count,
            "average_response_intensity": sum(r.get("response", {}).get("intensity", 0) for r in filtered_responses) / total_responses if total_responses > 0 else 0
        }
        
        return stats
    
    def simulate_response_effects(self, responses: List[Dict[str, Any]],
                               world_state: Dict[str, Any],
                               duration: int = 12) -> Dict[str, Any]:
        """
        模拟响应效果随时间的演变
        
        Args:
            responses: 响应列表
            world_state: 世界状态
            duration: 模拟持续时间（月）
            
        Returns:
            效果演变字典
        """
        # 初始化效果跟踪
        effects = {
            "economic_impact": [0.0] * duration,
            "political_impact": [0.0] * duration,
            "social_impact": [0.0] * duration,
            "environmental_impact": [0.0] * duration,
            "humanitarian_impact": [0.0] * duration
        }
        
        # 为每个响应计算效果
        for response in responses:
            response_type = response.get("response_type")
            intensity = response.get("intensity", 0.5)
            effectiveness = response.get("effectiveness", ResponseEffectiveness.MEDIUM)
            
            # 转换有效性为数值
            effectiveness_value = effectiveness.value
            
            # 不同响应类型的效果曲线
            if response_type in [ResponseType.ECONOMIC_STIMULUS, ResponseType.ECONOMIC_MEASURES]:
                # 经济刺激的效果曲线
                for month in range(duration):
                    if month < 3:
                        # 延迟效应
                        impact = 0.1 * intensity * (effectiveness_value + 1) / 5
                    elif month < 8:
                        # 上升期
                        impact = 0.8 * intensity * (effectiveness_value + 1) / 5
                    else:
                        # 衰减期
                        impact = 0.8 * intensity * (effectiveness_value + 1) / 5 * (1 - (month - 8) / 4)
                    
                    effects["economic_impact"][month] += impact
            
            elif response_type in [ResponseType.HUMANITARIAN_AID]:
                # 人道主义援助的效果曲线
                for month in range(duration):
                    if month < 2:
                        # 快速响应
                        impact = 1.0 * intensity * (effectiveness_value + 1) / 5
                    elif month < 6:
                        # 稳定期
                        impact = 0.7 * intensity * (effectiveness_value + 1) / 5
                    else:
                        # 逐渐减少
                        impact = 0.7 * intensity * (effectiveness_value + 1) / 5 * (1 - (month - 6) / 6)
                    
                    effects["humanitarian_impact"][month] += impact
                    effects["social_impact"][month] += impact * 0.5
            
            elif response_type in [ResponseType.MILITARY_INTERVENTION, ResponseType.POLICING_ACTION]:
                # 军事干预的效果曲线
                for month in range(duration):
                    if month < 1:
                        # 立即效果
                        security_impact = 0.9 * intensity * (effectiveness_value + 1) / 5
                        political_impact = -0.3 * intensity
                    elif month < 4:
                        # 短期效果
                        security_impact = 0.7 * intensity * (effectiveness_value + 1) / 5
                        political_impact = -0.2 * intensity
                    elif month < 8:
                        # 中期效果
                        security_impact = 0.5 * intensity * (effectiveness_value + 1) / 5
                        political_impact = -0.1 * intensity
                    else:
                        # 长期效果（可能引发反弹）
                        security_impact = 0.3 * intensity * (effectiveness_value + 1) / 5
                        political_impact = -0.05 * intensity
                    
                    effects["political_impact"][month] += political_impact
                    effects["social_impact"][month] += security_impact
            
            elif response_type in [ResponseType.PUBLIC_HEALTH_MEASURES]:
                # 公共卫生措施的效果曲线
                for month in range(duration):
                    if month < 1:
                        # 启动期
                        impact = 0.3 * intensity * (effectiveness_value + 1) / 5
                    elif month < 5:
                        # 上升期
                        impact = 1.0 * intensity * (effectiveness_value + 1) / 5
                    elif month < 9:
                        # 高峰期
                        impact = 0.8 * intensity * (effectiveness_value + 1) / 5
                    else:
                        # 衰减期
                        impact = 0.5 * intensity * (effectiveness_value + 1) / 5
                    
                    effects["humanitarian_impact"][month] += impact
                    effects["social_impact"][month] += impact * 0.7
            
            elif response_type in [ResponseType.TECHNOLOGICAL_ADAPTATION, ResponseType.INVESTMENT]:
                # 技术投资的效果曲线
                for month in range(duration):
                    # 延迟但持续增长的效果
                    impact = 0.2 * intensity * (effectiveness_value + 1) / 5 * (month / duration)
                    
                    effects["economic_impact"][month] += impact * 0.7
                    effects["environmental_impact"][month] += impact * 0.3
        
        # 确保所有影响在合理范围内
        for impact_type, values in effects.items():
            for i in range(duration):
                effects[impact_type][i] = min(max(values[i], -1.0), 1.0)
        
        return effects


# 测试代码
if __name__ == "__main__":
    # 导入必要的类和枚举
    from ..data.data_types import Event, EventType, EventSeverity, ResponseType, ResponseEffectiveness
    
    # 创建模拟配置
    config = {
        "country_capabilities": {},  # 使用默认配置
        "max_country_responses_per_event": 10,
        "max_organization_responses_per_event": 5
    }
    
    # 创建模型实例
    crisis_model = CrisisResponseModel(config)
    
    # 创建测试事件
    test_event = Event(
        id="test_event_001",
        type=EventType.CONFLICT,
        subtype="武装冲突",
        description="两国边界武装冲突",
        severity=EventSeverity.MAJOR,
        region="Middle_East",
        scope="regional",
        start_time=2023,
        end_time=None,
        probability=0.8,
        metadata={
            "participants": ["Country_A", "Country_B"],
            "casualties": 1000,
            "refugees": 50000
        }
    )
    
    # 创建模拟世界状态
    world_state = {
        "current_year": 2023,
        "global_stability": 0.6,
        "economic_health": 0.55,
        "technological_progress": 0.7,
        "climate_risk": 0.65
    }
    
    political_state = {
        "international_cooperation": 0.5,
        "major_alliances": [
            {"name": "Alliance_A", "members": ["United_States", "European_Union", "Japan"]},
            {"name": "Alliance_B", "members": ["China", "Russia"]}
        ],
        "conflict_zones": ["Middle_East", "Africa"]
    }
    
    economic_state = {
        "global_gdp_growth": 2.3,
        "inflation_rate": 3.5,
        "unemployment_rate": 5.2,
        "debt_levels": 0.7
    }
    
    print("\n===== 测试危机响应模型 =====")
    
    # 生成响应
    responses = crisis_model.generate_responses(
        test_event,
        world_state,
        political_state,
        economic_state,
        seed=42
    )
    
    print(f"\n生成了 {len(responses)} 个响应:")
    
    # 打印部分响应详情
    for i, response in enumerate(responses[:3]):  # 只打印前3个
        print(f"\n响应 {i+1}:")
        print(f"  行为者: {response.get('actor')} ({response.get('actor_type')})")
        print(f"  类型: {response.get('response_type')}")
        print(f"  描述: {response.get('description')}")
        print(f"  强度: {response.get('intensity'):.2f}")
        print(f"  及时性: {response.get('timeliness'):.2f}")
        print(f"  有效性: {response.get('effectiveness')}")
        
        # 根据行为者类型打印不同信息
        if response.get('actor_type') == "country":
            print(f"  公共支持: {response.get('public_support'):.2f}")
            print(f"  经济资源: {response.get('resources', {}).get('economic'):.4f} (GDP%)")
        else:
            print(f"  成员国支持: {response.get('member_support'):.2f}")
            print(f"  资金动员: {response.get('resources', {}).get('funding_billion_usd'):.2f} (十亿美元)")
    
    # 模拟响应效果
    print("\n===== 模拟响应效果 =====")
    effects = crisis_model.simulate_response_effects(responses, world_state, duration=6)
    
    print("\n月度效果演变 (前6个月):")
    print("月份  |  经济影响  |  政治影响  |  社会影响  |  人道主义影响")
    print("------|------------|------------|------------|------------")
    
    for month in range(6):
        print(f"{month+1:4d}   |  {effects['economic_impact'][month]:10.3f}  |  " 
              f"{effects['political_impact'][month]:10.3f}  |  {effects['social_impact'][month]:10.3f}  |  "
              f"{effects['humanitarian_impact'][month]:10.3f}")
    
    # 获取响应统计
    stats = crisis_model.get_response_statistics()
    print("\n===== 响应统计 =====")
    print(f"总响应数: {stats['total_responses']}")
    print("\n响应类型分布:")
    for r_type, count in stats['responses_by_type'].items():
        print(f"  {r_type}: {count}")
    
    print("\n危机响应模型测试完成!")