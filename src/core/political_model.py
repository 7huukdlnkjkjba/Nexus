#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
政治模型模块
负责模拟全球政治系统和国际关系的演化
"""

import logging
import numpy as np
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from ..utils.logger import get_logger

class PoliticalModel:
    """政治模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化政治模型
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("PoliticalModel")
        self.config = config
        
        # 模型参数
        self.volatility = config.get("political_volatility", 0.04)  # 政治波动性
        self.conflict_probability = config.get("conflict_probability", 0.015)  # 冲突概率
        self.alliance_probability = config.get("alliance_probability", 0.02)  # 联盟形成概率
        self.stability_shift_factor = config.get("stability_shift_factor", 0.03)  # 稳定性变化因子
        self.election_cycle_factor = config.get("election_cycle_factor", 0.05)  # 选举周期因子
        self.conflict_decay_rate = config.get("conflict_decay_rate", 0.1)  # 冲突衰减率
        
        # 主要国家和地区
        self.major_countries = config.get("major_countries", [
            "US", "China", "Russia", "UK", "France", "Germany", "Japan", 
            "India", "Brazil", "South Africa", "Australia", "Canada", 
            "South Korea", "Israel", "Saudi Arabia", "Iran", "North Korea"
        ])
        
        # 地区定义
        self.regions = config.get("regions", {
            "North America": ["US", "Canada"],
            "Europe": ["UK", "France", "Germany", "Italy", "Spain", "Poland"],
            "Asia-Pacific": ["China", "Japan", "India", "Australia", "South Korea"],
            "Middle East": ["Israel", "Saudi Arabia", "Iran", "Turkey"],
            "Africa": ["South Africa", "Egypt", "Nigeria"],
            "South America": ["Brazil", "Argentina", "Chile"]
        })
        
        # 冲突热点地区
        self.conflict_hotspots = config.get("conflict_hotspots", [
            "Taiwan Strait", "South China Sea", "Ukraine", "Middle East", 
            "Korean Peninsula", "India-Pakistan Border", "Baltic States"
        ])
        
        # 国际关系类型
        self.relationship_types = ["ally", "friendly", "neutral", "tense", "hostile", "at_war"]
        
        # 国家参数映射
        self.country_parameters = config.get("country_parameters", {})
        
        # 领域间影响权重
        self.cross_impact_weights = {
            "economic": config.get("economic_impact_weight", 0.4),
            "technological": config.get("technological_impact_weight", 0.25),
            "climate": config.get("climate_impact_weight", 0.2)
        }
        
        # 地区紧张度阈值
        self.tension_thresholds = {
            "low": 30,
            "medium": 60,
            "high": 80
        }
        
        self.logger.info("政治模型初始化完成")
    
    def evolve(self, political_state: Dict[str, Any],
              current_time: datetime,
              random_state: np.random.RandomState,
              external_factors: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        演化政治状态
        
        Args:
            political_state: 当前政治状态
            current_time: 当前时间
            random_state: 随机数生成器
            external_factors: 外部影响因子（如经济、技术、气候事件）
            
        Returns:
            更新后的政治状态
        """
        if external_factors is None:
            external_factors = {}
        
        # 复制状态以避免修改原始数据
        new_state = political_state.copy()
        
        # 演化国家稳定性
        if "countries" in new_state:
            for country in self.major_countries:
                if country in new_state["countries"]:
                    new_state["countries"][country] = self._evolve_country_stability(
                        country,
                        new_state["countries"][country],
                        random_state,
                        current_time,
                        external_factors
                    )
        
        # 演化国际关系
        if "relationships" in new_state:
            new_state["relationships"] = self._evolve_relationships(
                new_state["relationships"],
                new_state.get("countries", {}),
                random_state,
                external_factors
            )
        
        # 演化地区稳定性
        if "regions" in new_state:
            new_state["regions"] = self._evolve_region_stability(
                new_state["regions"],
                new_state.get("countries", {})
            )
        
        # 兼容旧格式：演化地区紧张度
        if "tensions" in new_state:
            new_state["tensions"] = self._evolve_tensions(new_state["tensions"], random_state)
        
        # 兼容旧格式：演化联盟
        if "alliances" in new_state:
            new_state["alliances"] = self._evolve_alliances(new_state["alliances"], new_state.get("tensions", {}), random_state)
        
        # 兼容旧格式：演化外交关系
        if "diplomacy" in new_state:
            new_state["diplomacy"] = self._evolve_diplomacy(
                new_state["diplomacy"],
                new_state.get("tensions", {}),
                new_state.get("alliances", []),
                random_state
            )
        
        # 更新全球政治指标
        if "global_indices" not in new_state:
            new_state["global_indices"] = {}
        new_state["global_indices"] = self._update_global_indices(new_state)
        
        # 可能发生政治事件
        new_events = self._generate_political_events(new_state, random_state, current_time, external_factors)
        if "events" not in new_state:
            new_state["events"] = []
        new_state["events"].extend(new_events)
        
        # 更新选举信息
        if "elections" not in new_state:
            new_state["elections"] = []
        new_state["elections"] = self._update_election_calendar(new_state["elections"], current_time, random_state)
        
        # 更新时间戳
        new_state["last_updated"] = current_time.isoformat()
        
        return new_state
    
    def _evolve_country_stability(self, country: str,
                                 country_data: Dict[str, Any],
                                 random_state: np.random.RandomState,
                                 current_time: datetime,
                                 external_factors: Dict[str, Any]) -> Dict[str, Any]:
        """
        演化国家政治稳定性
        
        Args:
            country: 国家名称
            country_data: 国家数据
            random_state: 随机数生成器
            current_time: 当前时间
            external_factors: 外部影响因子
            
        Returns:
            更新后的国家数据
        """
        # 复制国家数据
        new_data = country_data.copy()
        
        # 初始稳定性
        stability = new_data.get("stability", 70.0)
        
        # 随机波动
        random_shift = random_state.normal(0, self.volatility * 10)
        
        # 选举周期影响
        election_factor = 0.0
        # 简单模拟：如果今年有选举，稳定性波动加大
        if self._has_election_this_year(country, current_time):
            # 选举前稳定性可能下降，选举后可能上升（视选举结果而定）
            month = current_time.month
            if month < 6:  # 选举年上半年，可能存在不确定性
                election_factor = -self.election_cycle_factor * 10
            else:  # 选举年下半年，结果已出，稳定性可能恢复
                election_factor = random_state.choice([-self.election_cycle_factor, self.election_cycle_factor]) * 10
        
        # 经济因素影响
        economic_factor = external_factors.get("economic_impact", {}).get(country, 0.0)
        
        # 技术因素影响
        tech_factor = external_factors.get("technological_impact", {}).get(country, 0.0)
        
        # 气候事件影响
        climate_factor = external_factors.get("climate_impact", {}).get(country, 0.0)
        
        # 计算总变化
        total_shift = random_shift + election_factor + \
                     economic_factor * self.cross_impact_weights["economic"] + \
                     tech_factor * self.cross_impact_weights["technological"] + \
                     climate_factor * self.cross_impact_weights["climate"]
        
        # 国家特定因素调整
        country_factor = self._get_country_stability_factor(country)
        total_shift *= country_factor
        
        # 更新稳定性
        new_stability = stability + total_shift
        
        # 限制在0-100范围内
        new_stability = max(0, min(100, new_stability))
        
        # 更新政府支持率（假设与稳定性相关）
        support_rate = new_data.get("government_support", 60.0)
        support_change = random_state.normal(0, 2.0)  # 支持率变化有更大的随机性
        # 稳定性严重下降时，支持率可能大幅下降
        if total_shift < -5:
            support_change -= 5.0
        # 稳定性显著上升时，支持率可能上升
        elif total_shift > 5:
            support_change += 3.0
        
        new_support_rate = support_rate + support_change
        new_support_rate = max(0, min(100, new_support_rate))
        
        # 更新数据
        new_data["stability"] = new_stability
        new_data["government_support"] = new_support_rate
        
        # 更新国内冲突风险（稳定性越低，风险越高）
        new_data["internal_conflict_risk"] = max(0, min(100, 100 - new_stability * 0.8))
        
        # 更新威权指数（可以是配置的基础值加上随机波动）
        authoritarian_baseline = self.country_parameters.get(country, {}).get("authoritarian_baseline", 50.0)
        new_data["authoritarian_index"] = authoritarian_baseline + random_state.normal(0, 3.0)
        new_data["authoritarian_index"] = max(0, min(100, new_data["authoritarian_index"]))
        
        return new_data
    
    def _evolve_relationships(self, relationships: Dict[str, Dict[str, Any]],
                             countries: Dict[str, Dict[str, Any]],
                             random_state: np.random.RandomState,
                             external_factors: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        演化国际关系
        
        Args:
            relationships: 当前国际关系 {country1: {country2: relationship}}
            countries: 国家数据
            random_state: 随机数生成器
            external_factors: 外部影响因子
            
        Returns:
            更新后的国际关系
        """
        updated_relationships = {}
        
        # 为每个国家对创建关系
        for country1 in self.major_countries:
            if country1 not in updated_relationships:
                updated_relationships[country1] = {}
            
            for country2 in self.major_countries:
                if country1 == country2:  # 国家不会与自己有关系
                    continue
                
                # 获取现有关系或创建默认关系
                if country1 in relationships and country2 in relationships[country1]:
                    current_relationship = relationships[country1][country2].copy()
                else:
                    # 默认中性关系
                    current_relationship = {
                        "type": "neutral",
                        "trust_score": 50.0,
                        "conflict_level": 10.0,
                        "trade_integration": 30.0,
                        "military_alliance": False,
                        "diplomatic_ties": "normal"
                    }
                
                # 演化信任分数
                trust_score = current_relationship["trust_score"]
                
                # 基础信任变化（关系类型影响变化方向）
                type_factor = {
                    "ally": 1.0,
                    "friendly": 0.5,
                    "neutral": 0.0,
                    "tense": -0.5,
                    "hostile": -1.0,
                    "at_war": -2.0
                }.get(current_relationship["type"], 0.0)
                
                # 随机波动
                random_factor = random_state.normal(0, 5.0)
                
                # 经济整合影响
                trade_impact = current_relationship.get("trade_integration", 30.0) * 0.1
                
                # 共同敌人影响（简化版：如果都与第三国处于敌对关系，可能增加信任）
                common_enemy_bonus = 0.0
                if "at_war" not in [current_relationship["type"]]:
                    for country3 in self.major_countries:
                        if country3 != country1 and country3 != country2:
                            rel1_3 = relationships.get(country1, {}).get(country3, {})
                            rel2_3 = relationships.get(country2, {}).get(country3, {})
                            if rel1_3.get("type") in ["hostile", "at_war"] and \
                               rel2_3.get("type") in ["hostile", "at_war"]:
                                common_enemy_bonus += 2.0
                                break
                
                # 国内稳定性影响
                stability1 = countries.get(country1, {}).get("stability", 70.0)
                stability2 = countries.get(country2, {}).get("stability", 70.0)
                stability_impact = (stability1 + stability2) / 100.0 * 2.0 - 2.0
                
                # 计算新信任分数
                new_trust_score = trust_score + type_factor + random_factor + trade_impact + \
                                 common_enemy_bonus + stability_impact
                new_trust_score = max(0, min(100, new_trust_score))
                
                # 更新关系类型
                old_type = current_relationship["type"]
                new_type = self._determine_relationship_type(new_trust_score, old_type)
                
                # 更新冲突水平
                conflict_level = current_relationship["conflict_level"]
                # 关系越差，冲突水平可能越高
                conflict_change = random_state.normal(0, 5.0)
                if new_type in ["hostile", "at_war"]:
                    conflict_change += 5.0
                elif new_type in ["ally", "friendly"]:
                    conflict_change -= 3.0
                
                new_conflict_level = conflict_level + conflict_change
                new_conflict_level = max(0, min(100, new_conflict_level))
                
                # 更新贸易整合
                trade_integration = current_relationship.get("trade_integration", 30.0)
                # 关系改善可能促进贸易整合
                trade_change = random_state.normal(0, 2.0)
                if new_type in ["ally", "friendly"]:
                    trade_change += 1.0
                elif new_type in ["hostile", "at_war"]:
                    trade_change -= 5.0
                
                new_trade_integration = trade_integration + trade_change
                new_trade_integration = max(0, min(100, new_trade_integration))
                
                # 更新军事联盟
                military_alliance = current_relationship.get("military_alliance", False)
                # 盟友关系更可能形成军事联盟
                if new_type == "ally" and not military_alliance:
                    if random_state.random() < 0.1:  # 10%概率形成军事联盟
                        military_alliance = True
                elif new_type != "ally" and military_alliance:
                    if random_state.random() < 0.05:  # 5%概率解除军事联盟
                        military_alliance = False
                
                # 更新外交关系
                diplomatic_ties = current_relationship.get("diplomatic_ties", "normal")
                if new_type == "at_war":
                    diplomatic_ties = "severed"
                elif new_type == "hostile" and diplomatic_ties != "severed":
                    diplomatic_ties = "minimal"
                elif new_type in ["ally", "friendly"] and diplomatic_ties != "normal":
                    diplomatic_ties = "enhanced"
                elif new_type == "neutral" and diplomatic_ties not in ["normal", "minimal"]:
                    diplomatic_ties = "normal"
                
                # 更新关系数据
                updated_relationships[country1][country2] = {
                    "type": new_type,
                    "trust_score": new_trust_score,
                    "conflict_level": new_conflict_level,
                    "trade_integration": new_trade_integration,
                    "military_alliance": military_alliance,
                    "diplomatic_ties": diplomatic_ties,
                    "last_update": datetime.now().isoformat()
                }
        
        # 检查是否形成新的联盟或冲突
        self._check_for_alliances_and_conflicts(updated_relationships, random_state)
        
        return updated_relationships
    
    def _evolve_region_stability(self, regions: Dict[str, Dict[str, Any]],
                                countries: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        演化地区稳定性
        
        Args:
            regions: 地区数据
            countries: 国家数据
            
        Returns:
            更新后的地区数据
        """
        updated_regions = {}
        
        for region_name, region_countries in self.regions.items():
            # 获取地区内国家的平均稳定性
            stabilities = []
            for country in region_countries:
                if country in countries:
                    stabilities.append(countries[country].get("stability", 70.0))
            
            if stabilities:
                avg_stability = sum(stabilities) / len(stabilities)
            else:
                avg_stability = 70.0
            
            # 计算地区冲突风险（与稳定性相反）
            conflict_risk = max(0, min(100, 100 - avg_stability))
            
            # 计算地区一体化水平（基于内部关系）
            integration = regions.get(region_name, {}).get("integration", 50.0)
            
            updated_regions[region_name] = {
                "stability": avg_stability,
                "conflict_risk": conflict_risk,
                "integration": integration,
                "countries": region_countries,
                "last_update": datetime.now().isoformat()
            }
        
        return updated_regions
    
    def _update_global_indices(self, political_state: Dict[str, Any]) -> Dict[str, float]:
        """
        更新全球政治指标
        
        Args:
            political_state: 政治状态
            
        Returns:
            全球政治指标
        """
        indices = {}
        
        # 计算全球平均稳定性
        countries = political_state.get("countries", {})
        stabilities = [data.get("stability", 70.0) for data in countries.values()]
        if stabilities:
            indices["global_stability"] = sum(stabilities) / len(stabilities)
        else:
            indices["global_stability"] = 70.0
        
        # 计算全球冲突水平
        relationships = political_state.get("relationships", {})
        conflicts = []
        for country1, rels in relationships.items():
            for country2, rel_data in rels.items():
                if country1 != country2:  # 避免重复计算
                    conflicts.append(rel_data.get("conflict_level", 10.0))
        
        if conflicts:
            indices["global_conflict"] = sum(conflicts) / len(conflicts)
        else:
            indices["global_conflict"] = 10.0
        
        # 计算多极化指数（基于主要大国之间的关系分布）
        major_powers = ["US", "China", "Russia", "EU", "India"]
        power_relationships = []
        for country1 in major_powers:
            for country2 in major_powers:
                if country1 != country2 and country1 in relationships and country2 in relationships[country1]:
                    rel_type = relationships[country1][country2].get("type", "neutral")
                    # 将关系类型转换为数值
                    type_value = {
                        "ally": 5.0,
                        "friendly": 3.0,
                        "neutral": 0.0,
                        "tense": -3.0,
                        "hostile": -5.0,
                        "at_war": -10.0
                    }.get(rel_type, 0.0)
                    power_relationships.append(type_value)
        
        if power_relationships:
            # 关系越分散（标准差越大），多极化程度越高
            if len(power_relationships) > 1:
                indices["multipolarity"] = np.std(power_relationships) * 10
            else:
                indices["multipolarity"] = 0.0
        else:
            indices["multipolarity"] = 0.0
        
        # 限制范围
        indices["multipolarity"] = max(0, min(100, indices["multipolarity"]))
        
        return indices
    
    def _generate_political_events(self, political_state: Dict[str, Any],
                                 random_state: np.random.RandomState,
                                 current_time: datetime,
                                 external_factors: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成政治事件
        
        Args:
            political_state: 政治状态
            random_state: 随机数生成器
            current_time: 当前时间
            external_factors: 外部影响因子
            
        Returns:
            政治事件列表
        """
        events = []
        
        # 事件模板
        event_templates = {
            "election": [
                "{country}举行大选，{party}获胜",
                "{country}总统选举结束，{party}候选人成功连任"
            ],
            "protest": [
                "{country}爆发大规模抗议活动，民众要求{demand}",
                "{country}多个城市出现示威游行，抗议{issue}"
            ],
            "government_change": [
                "{country}发生政府更迭，{new_leader}上台",
                "{country}内阁重组，{new_official}被任命为{position}"
            ],
            "conflict": [
                "{country1}与{country2}在{location}发生军事冲突",
                "{region}地区冲突升级，{country}介入"
            ],
            "alliance": [
                "{country1}与{country2}签署新的战略合作协议",
                "{region}国家宣布组建新的地区联盟"
            ],
            "crisis": [
                "{country}面临严重政治危机，{details}",
                "{country}政府因{scandal}陷入信任危机"
            ],
            "reforms": [
                "{country}实施重大政治改革，包括{reforms}",
                "{country}宣布经济政策改革，旨在解决{issue}"
            ]
        }
        
        # 政党和领导人示例
        parties = {
            "US": ["民主党", "共和党", "独立党"],
            "China": ["共产党"],
            "Russia": ["统一俄罗斯党", "俄罗斯联邦共产党"],
            "UK": ["保守党", "工党", "自由民主党"],
            "France": ["共和国前进党", "国民联盟", "社会党"],
            "Germany": ["联盟党", "社会民主党", "绿党", "选择党"],
            "Japan": ["自民党", "立宪民主党", "公明党"]
        }
        
        # 生成国家层面事件
        for country in self.major_countries:
            # 基于国家稳定性调整事件概率
            country_stability = political_state.get("countries", {}).get(country, {}).get("stability", 70.0)
            event_probability = 0.1 * (100 - country_stability) / 100  # 稳定性越低，事件概率越高
            
            if random_state.random() < event_probability:
                # 选择事件类型
                event_types = list(event_templates.keys())
                # 低稳定性国家更可能发生冲突和抗议
                weights = np.ones(len(event_types))
                if country_stability < 40:
                    weights[event_types.index("protest")] = 2.0
                    weights[event_types.index("crisis")] = 1.5
                event_type = random_state.choice(event_types, p=weights/weights.sum())
                
                # 生成事件描述
                if event_type == "election":
                    country_parties = parties.get(country, ["执政党", "反对党"])
                    winning_party = random_state.choice(country_parties)
                    template = random_state.choice(event_templates[event_type])
                    description = template.format(country=country, party=winning_party)
                elif event_type == "protest":
                    demands = ["政府改革", "经济改善", "政治自由", "反腐败", "社会福利增加"]
                    demand = random_state.choice(demands)
                    template = random_state.choice(event_templates[event_type])
                    description = template.format(country=country, demand=demand, issue=demand)
                elif event_type == "government_change":
                    positions = ["总统", "总理", "首相", "国务卿", "外交部长"]
                    position = random_state.choice(positions)
                    template = random_state.choice(event_templates[event_type])
                    description = template.format(country=country, new_leader="新领导人", new_official="新官员", position=position)
                elif event_type == "crisis":
                    details = ["领导人健康问题", "政府腐败曝光", "政治分裂加剧"]
                    detail = random_state.choice(details)
                    scandals = ["腐败", "选举舞弊", "滥用权力"]
                    scandal = random_state.choice(scandals)
                    template = random_state.choice(event_templates[event_type])
                    description = template.format(country=country, details=detail, scandal=scandal)
                elif event_type == "reforms":
                    reforms = ["选举制度改革", "反腐败法案", "宪法修正案"]
                    reform_list = random_state.choice(reforms, size=min(2, len(reforms)), replace=False)
                    reform_text = "和".join(reform_list)
                    issues = ["经济不平等", "政府效率低下", "社会分裂"]
                    issue = random_state.choice(issues)
                    template = random_state.choice(event_templates[event_type])
                    description = template.format(country=country, reforms=reform_text, issue=issue)
                else:
                    # 跳过需要两国参与的事件类型
                    continue
                
                # 创建事件
                event = {
                    "id": f"{event_type}_{country}_{int(current_time.timestamp())}_{random_state.randint(1000, 9999)}",
                    "type": event_type,
                    "description": description,
                    "country": country,
                    "timestamp": current_time.isoformat(),
                    "severity": random_state.randint(1, 6),
                    "potential_impact": self._calculate_event_impact(event_type, country)
                }
                
                events.append(event)
                self.logger.info(f"政治事件: {description} (国家: {country}, 严重度: {event['severity']})")
        
        # 生成国际事件
        # 检查冲突
        if random_state.random() < self.conflict_probability:
            # 选择冲突热点
            hotspot = random_state.choice(self.conflict_hotspots)
            
            # 根据热点选择相关国家
            hotspot_countries = {
                "Taiwan Strait": ["US", "China"],
                "South China Sea": ["China", "Philippines", "Vietnam", "US"],
                "Ukraine": ["Russia", "Ukraine", "US", "EU"],
                "Middle East": ["Israel", "Iran", "Saudi Arabia", "US"],
                "Korean Peninsula": ["North Korea", "South Korea", "US", "China"],
                "India-Pakistan Border": ["India", "Pakistan"],
                "Baltic States": ["Russia", "NATO", "Estonia", "Latvia", "Lithuania"]
            }
            
            relevant_countries = hotspot_countries.get(hotspot, ["Country A", "Country B"])
            if len(relevant_countries) >= 2:
                country1, country2 = random_state.choice(relevant_countries, size=2, replace=False)
                
                template = random_state.choice(event_templates["conflict"])
                description = template.format(country1=country1, country2=country2, location=hotspot, region=hotspot)
                
                event = {
                    "id": f"conflict_{int(current_time.timestamp())}_{random_state.randint(1000, 9999)}",
                    "type": "conflict",
                    "description": description,
                    "countries": [country1, country2],
                    "location": hotspot,
                    "timestamp": current_time.isoformat(),
                    "severity": random_state.randint(3, 6),  # 冲突通常比较严重
                    "potential_impact": self._calculate_conflict_impact(country1, country2, hotspot)
                }
                
                events.append(event)
                self.logger.info(f"国际冲突: {description} (严重度: {event['severity']})")
        
        # 检查联盟形成
        if random_state.random() < self.alliance_probability:
            # 选择地区
            region = random_state.choice(list(self.regions.keys()))
            region_countries = self.regions[region]
            
            # 选择2-3个国家
            if len(region_countries) >= 2:
                num_countries = random_state.randint(2, min(4, len(region_countries) + 1))
                alliance_countries = random_state.choice(region_countries, size=num_countries, replace=False)
                
                template = random_state.choice(event_templates["alliance"])
                if num_countries == 2:
                    description = template.format(country1=alliance_countries[0], 
                                                country2=alliance_countries[1], 
                                                region=region)
                else:
                    description = template.format(region=region)
                
                event = {
                    "id": f"alliance_{int(current_time.timestamp())}_{random_state.randint(1000, 9999)}",
                    "type": "alliance",
                    "description": description,
                    "countries": alliance_countries.tolist(),
                    "region": region,
                    "timestamp": current_time.isoformat(),
                    "severity": random_state.randint(1, 4),
                    "potential_impact": self._calculate_alliance_impact(alliance_countries)
                }
                
                events.append(event)
                self.logger.info(f"联盟形成: {description} (严重度: {event['severity']})")
        
        return events
    
    def _update_election_calendar(self, current_elections: List[Dict[str, Any]],
                                current_time: datetime,
                                random_state: np.random.RandomState) -> List[Dict[str, Any]]:
        """
        更新选举日历
        
        Args:
            current_elections: 当前选举计划
            current_time: 当前时间
            random_state: 随机数生成器
            
        Returns:
            更新后的选举日历
        """
        # 主要国家的选举周期
        election_cycles = {
            "US": {"presidential": 4, "congressional": 2},
            "China": {"leadership": 10},
            "Russia": {"presidential": 6, "parliamentary": 5},
            "UK": {"general": 5},
            "France": {"presidential": 5, "parliamentary": 5},
            "Germany": {"federal": 4},
            "Japan": {"general": 4, "upper_house": 3},
            "India": {"general": 5},
            "Brazil": {"presidential": 4, "legislative": 4},
            "South Africa": {"general": 5}
        }
        
        updated_elections = []
        current_year = current_time.year
        
        # 检查现有选举记录，移除过去的选举
        for election in current_elections:
            election_date = datetime.fromisoformat(election["date"])
            if election_date >= current_time:
                updated_elections.append(election)
        
        # 为未来3年生成选举计划
        for year in range(current_year, current_year + 4):
            for country, cycles in election_cycles.items():
                for election_type, cycle in cycles.items():
                    # 简化逻辑：假设选举在固定年份举行
                    # 实际中需要更复杂的规则，但这里做简化处理
                    if (year - 2020) % cycle == 0 or (year - 2021) % cycle == 0:
                        # 检查是否已经有该选举的记录
                        exists = any(e["country"] == country and 
                                   e["type"] == election_type and 
                                   datetime.fromisoformat(e["date"]).year == year 
                                   for e in updated_elections)
                        
                        if not exists:
                            # 生成选举日期（随机月份）
                            month = random_state.randint(1, 13)
                            day = random_state.randint(1, 29)  # 简化处理
                            election_date = datetime(year, month, day)
                            
                            if election_date >= current_time:
                                updated_elections.append({
                                    "id": f"election_{country}_{election_type}_{year}",
                                    "country": country,
                                    "type": election_type,
                                    "date": election_date.isoformat(),
                                    "status": "scheduled",
                                    "uncertainty": random_state.uniform(0, 0.2)
                                })
        
        return updated_elections
    
    def _determine_relationship_type(self, trust_score: float, current_type: str) -> str:
        """
        根据信任分数确定关系类型
        
        Args:
            trust_score: 信任分数
            current_type: 当前关系类型（用于添加连续性）
            
        Returns:
            新的关系类型
        """
        # 关系类型阈值
        thresholds = {
            "ally": 80.0,
            "friendly": 60.0,
            "neutral": 40.0,
            "tense": 20.0,
            "hostile": 0.0
        }
        
        # 首先根据信任分数确定基本类型
        if trust_score >= thresholds["ally"]:
            base_type = "ally"
        elif trust_score >= thresholds["friendly"]:
            base_type = "friendly"
        elif trust_score >= thresholds["neutral"]:
            base_type = "neutral"
        elif trust_score >= thresholds["tense"]:
            base_type = "tense"
        else:
            base_type = "hostile"
        
        # 添加连续性：关系不容易急剧变化
        # 如果当前处于战争状态，只有信任显著改善才会改变
        if current_type == "at_war":
            if trust_score > thresholds["hostile"] + 10:
                return "hostile"
            return "at_war"
        
        # 其他情况下允许更平滑的过渡
        return base_type
    
    def _check_for_alliances_and_conflicts(self, relationships: Dict[str, Dict[str, Any]],
                                         random_state: np.random.RandomState):
        """
        检查是否形成新的联盟或升级为战争
        
        Args:
            relationships: 国际关系
            random_state: 随机数生成器
        """
        for country1, rels in relationships.items():
            for country2, rel_data in rels.items():
                if country1 == country2:
                    continue
                
                # 敌对关系可能升级为战争
                if rel_data["type"] == "hostile" and rel_data["conflict_level"] > 80:
                    if random_state.random() < 0.05:  # 5%概率升级为战争
                        relationships[country1][country2]["type"] = "at_war"
                        relationships[country1][country2]["conflict_level"] = 100
                        self.logger.warning(f"战争爆发: {country1} 与 {country2} 处于战争状态")
    
    def _calculate_event_impact(self, event_type: str, country: str) -> Dict[str, float]:
        """
        计算事件影响
        
        Args:
            event_type: 事件类型
            country: 国家
            
        Returns:
            影响程度
        """
        impact = {
            "domestic_politics": 0.0,
            "international_relations": 0.0,
            "economic": 0.0,
            "stability": 0.0
        }
        
        # 不同事件类型的影响权重
        if event_type == "election":
            impact["domestic_politics"] = 80.0
            impact["stability"] = 50.0
            impact["international_relations"] = 40.0
            impact["economic"] = 30.0
        elif event_type == "protest":
            impact["domestic_politics"] = 60.0
            impact["stability"] = 70.0
            impact["economic"] = 40.0
            impact["international_relations"] = 20.0
        elif event_type == "government_change":
            impact["domestic_politics"] = 70.0
            impact["stability"] = 60.0
            impact["international_relations"] = 50.0
            impact["economic"] = 40.0
        elif event_type == "crisis":
            impact["domestic_politics"] = 90.0
            impact["stability"] = 85.0
            impact["international_relations"] = 60.0
            impact["economic"] = 70.0
        elif event_type == "reforms":
            impact["domestic_politics"] = 75.0
            impact["stability"] = 65.0
            impact["economic"] = 80.0
            impact["international_relations"] = 30.0
        
        # 大国事件影响更大
        if country in ["US", "China", "Russia", "EU"]:
            for key in impact:
                impact[key] *= 1.5
        
        return impact
    
    def _calculate_conflict_impact(self, country1: str, country2: str, location: str) -> Dict[str, float]:
        """
        计算冲突影响
        
        Args:
            country1: 国家1
            country2: 国家2
            location: 冲突地点
            
        Returns:
            影响程度
        """
        impact = {
            "international_relations": 90.0,
            "economic": 70.0,
            "regional_stability": 85.0,
            "global_security": 60.0
        }
        
        # 核国家之间的冲突影响更大
        nuclear_countries = ["US", "China", "Russia", "UK", "France", "India", "Pakistan", "North Korea"]
        if country1 in nuclear_countries and country2 in nuclear_countries:
            for key in impact:
                impact[key] *= 1.8
        
        # 重要经济区域的冲突影响更大
        economic_hotspots = ["Taiwan Strait", "South China Sea", "Middle East"]
        if location in economic_hotspots:
            impact["economic"] *= 1.5
        
        return impact
    
    def _calculate_alliance_impact(self, countries: List[str]) -> Dict[str, float]:
        """
        计算联盟影响
        
        Args:
            countries: 联盟国家
            
        Returns:
            影响程度
        """
        impact = {
            "international_relations": 60.0,
            "regional_stability": 40.0,
            "economic_integration": 50.0,
            "military_cooperation": 70.0
        }
        
        # 大国参与的联盟影响更大
        major_powers = ["US", "China", "Russia", "EU"]
        power_count = sum(1 for country in countries if country in major_powers)
        
        if power_count >= 2:
            for key in impact:
                impact[key] *= 1.6
        elif power_count == 1:
            for key in impact:
                impact[key] *= 1.3
        
        return impact
    
    def _has_election_this_year(self, country: str, current_time: datetime) -> bool:
        """
        判断国家今年是否有选举
        
        Args:
            country: 国家名称
            current_time: 当前时间
            
        Returns:
            是否有选举
        """
        # 简化的选举年份判断
        election_years = {
            "US": {2020, 2022, 2024, 2026, 2028},
            "UK": {2019, 2024, 2029},
            "France": {2022, 2027},
            "Germany": {2021, 2025},
            "Japan": {2021, 2025}
        }
        
        country_years = election_years.get(country, set())
        return current_time.year in country_years
    
    def _get_country_stability_factor(self, country: str) -> float:
        """
        获取国家稳定性因子
        
        Args:
            country: 国家名称
            
        Returns:
            稳定性因子
        """
        # 从配置中获取国家特定的稳定性因子
        custom_factor = self.country_parameters.get(country, {}).get("stability_factor")
        if custom_factor is not None:
            return custom_factor
        
        # 预定义一些国家的稳定性因子
        stability_factors = {
            "US": 1.0,
            "China": 0.7,  # 较低的波动性
            "Russia": 1.3,  # 较高的波动性
            "UK": 0.9,
            "France": 1.1,
            "Germany": 0.8,
            "Japan": 0.7,
            "India": 1.2,
            "Brazil": 1.4,
            "South Africa": 1.5,
            "North Korea": 0.5,  # 极低的波动性
            "Iran": 1.2
        }
        
        return stability_factors.get(country, 1.0)
    
    def calculate_global_risk_index(self, political_state: Dict[str, Any]) -> float:
        """
        计算全球政治风险指数
        
        Args:
            political_state: 政治状态
            
        Returns:
            全球政治风险指数 (0-100)
        """
        # 基础风险指数基于全球稳定性
        global_stability = political_state.get("global_indices", {}).get("global_stability", 70.0)
        base_risk = 100 - global_stability
        
        # 考虑战争和严重冲突
        war_count = 0
        severe_conflict_count = 0
        relationships = political_state.get("relationships", {})
        
        for country1, rels in relationships.items():
            for country2, rel_data in rels.items():
                if country1 != country2:
                    if rel_data.get("type") == "at_war":
                        war_count += 1
                    elif rel_data.get("conflict_level", 0) > 80:
                        severe_conflict_count += 1
        
        # 战争和严重冲突大幅增加风险
        conflict_risk = war_count * 20 + severe_conflict_count * 10
        
        # 考虑地区冲突热点
        region_risk = 0
        regions = political_state.get("regions", {})
        for region_data in regions.values():
            if region_data.get("conflict_risk", 0) > 70:
                region_risk += 5
        
        # 计算总风险指数
        total_risk = base_risk + conflict_risk + region_risk
        
        # 限制在0-100范围内
        return max(0, min(100, total_risk))
    
    def _evolve_tensions(self, tensions: Dict[str, float], 
                        random_state: np.random.RandomState) -> Dict[str, float]:
        """
        演化地区紧张度（兼容旧格式）
        
        Args:
            tensions: 地区紧张度数据（0-100）
            random_state: 随机数生成器
            
        Returns:
            更新后的地区紧张度
        """
        updated_tensions = {}
        
        for region, tension in tensions.items():
            # 随机波动
            random_change = random_state.normal(0, self.volatility * 10)  # 紧张度波动
            
            # 均值回归：紧张度趋向中等水平（50）
            mean_reversion = self.conflict_decay_rate * (50 - tension)
            
            # 计算新的紧张度
            new_tension = tension + random_change + mean_reversion
            
            # 限制在0-100范围内
            new_tension = max(0, min(100, new_tension))
            
            updated_tensions[region] = new_tension
        
        return updated_tensions
    
    def _evolve_alliances(self, alliances: List[str], 
                         tensions: Dict[str, float],
                         random_state: np.random.RandomState) -> List[str]:
        """
        演化联盟关系（兼容旧格式）
        
        Args:
            alliances: 联盟列表
            tensions: 地区紧张度
            random_state: 随机数生成器
            
        Returns:
            更新后的联盟列表
        """
        updated_alliances = alliances.copy()
        
        # 可能分裂现有联盟
        if updated_alliances:
            for alliance in alliances:
                # 联盟分裂的概率取决于地区紧张度
                split_probability = 0.02  # 基础分裂概率
                
                # 检查是否有高紧张度地区可能影响这个联盟
                high_tension_areas = [r for r, t in tensions.items() if t > self.tension_thresholds["high"]]
                if high_tension_areas:
                    # 高紧张度地区增加联盟分裂概率
                    split_probability += 0.05 * len(high_tension_areas)
                
                # 决定是否分裂联盟
                if random_state.random() < split_probability:
                    # 移除旧联盟
                    updated_alliances.remove(alliance)
                    
                    # 创建两个新联盟（简化处理）
                    new_alliance1 = f"{alliance}_A_{int(datetime.now().timestamp())}"
                    new_alliance2 = f"{alliance}_B_{int(datetime.now().timestamp())}"
                    updated_alliances.extend([new_alliance1, new_alliance2])
                    
                    self.logger.info(f"联盟 {alliance} 分裂为 {new_alliance1} 和 {new_alliance2}")
        
        # 可能形成新联盟
        if random_state.random() < 0.05:  # 5%概率形成新联盟
            new_alliance = f"Alliance_{int(datetime.now().timestamp())}"
            updated_alliances.append(new_alliance)
            self.logger.info(f"形成新联盟: {new_alliance}")
        
        return updated_alliances
    
    def _evolve_diplomacy(self, diplomacy: Dict[str, Dict[str, float]],
                         tensions: Dict[str, float],
                         alliances: List[str],
                         random_state: np.random.RandomState) -> Dict[str, Dict[str, float]]:
        """
        演化外交关系
        
        Args:
            diplomacy: 外交关系矩阵
            tensions: 地区紧张度
            alliances: 联盟列表
            random_state: 随机数生成器
            
        Returns:
            更新后的外交关系矩阵
        """
        updated_diplomacy = {}
        
        # 复制现有外交关系
        for country1, rels in diplomacy.items():
            updated_diplomacy[country1] = {}
            for country2, relation in rels.items():
                # 外交关系随机波动
                change = random_state.normal(0, self.volatility * 10)
                
                # 地区紧张度影响外交关系
                region_impact = 0
                for region, tension in tensions.items():
                    # 简化处理：假设地区紧张度影响该地区所有国家的外交关系
                    if tension > self.tension_thresholds["high"]:
                        region_impact -= 0.5  # 高紧张度地区负面影响外交关系
                
                # 联盟影响外交关系
                alliance_impact = 0
                for alliance in alliances:
                    # 简化处理：如果两国在同一联盟，改善外交关系
                    if country1 in alliance and country2 in alliance:
                        alliance_impact += 1.0
                
                # 计算新的外交关系值
                new_relation = relation + change + region_impact + alliance_impact
                
                # 限制在-100到100范围内
                new_relation = max(-100, min(100, new_relation))
                
                updated_diplomacy[country1][country2] = new_relation
        
        return updated_diplomacy
    
    def _evolve_regimes(self, regimes: Dict[str, Dict[str, Any]],
                       random_state: np.random.RandomState,
                       current_time: datetime) -> Dict[str, Dict[str, Any]]:
        """
        演化政治制度
        
        Args:
            regimes: 各国政治制度信息
            random_state: 随机数生成器
            current_time: 当前时间
            
        Returns:
            更新后的政治制度信息
        """
        updated_regimes = {}
        
        for country, regime_info in regimes.items():
            updated_regimes[country] = regime_info.copy()
            
            # 检查是否举行选举
            last_election = datetime.fromisoformat(regime_info.get("last_election", current_time.isoformat()))
            election_frequency = self.country_parameters.get(country, {}).get("election_cycle", self.election_cycle)
            
            # 计算距离上次选举的时间
            years_since_election = (current_time - last_election).days / 365.25
            
            # 如果是民主国家且到了选举时间
            if regime_info.get("type") == "democracy" and years_since_election >= election_frequency:
                # 举行选举
                new_government = self._hold_election(country, regime_info, random_state, current_time)
                updated_regimes[country] = new_government
                
                self.logger.info(f"{country} 举行选举，新政府: {new_government['government']}")
            
            # 演化政治稳定性
            stability = regime_info.get("stability", 50)
            
            # 随机波动
            random_change = random_state.normal(0, self.volatility * 10)
            
            # 均值回归
            mean_reversion = 0.1 * (70 - stability)  # 趋向较高的稳定性
            
            # 新政府初期通常更稳定
            if regime_info.get("last_election") == regime_info.get("established_date"):
                mean_reversion += 2.0  # 新政府初期稳定性增加
            
            new_stability = stability + random_change + mean_reversion
            new_stability = max(0, min(100, new_stability))
            
            updated_regimes[country]["stability"] = new_stability
        
        return updated_regimes
    
    def _hold_election(self, country: str, 
                      current_regime: Dict[str, Any],
                      random_state: np.random.RandomState,
                      current_time: datetime) -> Dict[str, Any]:
        """
        举行选举
        
        Args:
            country: 国家名称
            current_regime: 当前政权信息
            random_state: 随机数生成器
            current_time: 当前时间
            
        Returns:
            新政权信息
        """
        # 获取可能的政党/候选人
        possible_governments = self.country_parameters.get(country, {}).get("political_parties", 
                                                                           ["Party A", "Party B", "Party C"])
        
        # 执政党有一定优势
        current_government = current_regime.get("government", possible_governments[0])
        
        # 计算各政党的支持率
        support_rates = {}
        for party in possible_governments:
            # 基础支持率
            base_support = random_state.uniform(0.2, 0.4)
            
            # 执政党优势
            if party == current_government:
                base_support += 0.1  # 现任政府额外10%支持率
            
            support_rates[party] = base_support
        
        # 归一化支持率
        total_support = sum(support_rates.values())
        for party in support_rates:
            support_rates[party] /= total_support
        
        # 选择获胜政党
        parties = list(support_rates.keys())
        probabilities = list(support_rates.values())
        new_government = random_state.choice(parties, p=probabilities)
        
        # 创建新政权信息
        new_regime = current_regime.copy()
        new_regime["government"] = new_government
        new_regime["last_election"] = current_time.isoformat()
        new_regime["established_date"] = current_time.isoformat()
        new_regime["stability"] = random_state.uniform(60, 80)  # 新政府初期稳定性较高
        
        return new_regime
    
    def _evolve_events(self, events: List[Dict[str, Any]],
                      current_time: datetime) -> List[Dict[str, Any]]:
        """
        演化政治事件（更新状态、移除过期事件等）
        
        Args:
            events: 政治事件列表
            current_time: 当前时间
            
        Returns:
            更新后的事件列表
        """
        updated_events = []
        
        for event in events:
            # 检查事件是否仍然活跃
            if "end_date" in event:
                end_date = datetime.fromisoformat(event["end_date"])
                if current_time > end_date:
                    # 事件已结束，标记为历史事件
                    event["active"] = False
                else:
                    event["active"] = True
            
            updated_events.append(event)
        
        # 只保留最近的100个事件
        if len(updated_events) > 100:
            # 按时间倒序排序，保留最新的
            updated_events.sort(key=lambda e: e.get("start_date", ""), reverse=True)
            updated_events = updated_events[:100]
        
        return updated_events
    
    def _generate_random_event(self, random_state: np.random.RandomState,
                              current_time: datetime) -> Optional[Dict[str, Any]]:
        """
        生成随机政治事件
        
        Args:
            random_state: 随机数生成器
            current_time: 当前时间
            
        Returns:
            政治事件，如果没有生成则返回None
        """
        # 10%概率生成事件
        if random_state.random() > 0.1:
            return None
        
        # 可能的事件类型
        event_types = [
            "election",
            "protest",
            "coup_attempt",
            "diplomatic_crisis",
            "treaty_signing",
            "leadership_change",
            "sanctions_imposed",
            "terrorist_attack",
            "natural_disaster_response",
            "scandal_revelation"
        ]
        
        # 随机选择事件类型
        event_type = random_state.choice(event_types)
        
        # 随机选择地区/国家
        regions = ["Middle East", "Europe", "Asia", "Africa", "Americas"]
        countries = ["US", "China", "Russia", "EU", "India", "Brazil", "Japan", "Germany"]
        
        region = random_state.choice(regions)
        primary_country = random_state.choice(countries)
        
        # 事件持续时间（天数）
        duration = random_state.randint(1, 365)
        end_date = current_time + timedelta(days=duration)
        
        # 事件严重程度（1-5）
        severity = random_state.randint(1, 6)
        
        # 创建事件
        event = {
            "id": f"event_{int(current_time.timestamp())}_{random_state.randint(1000, 9999)}",
            "type": event_type,
            "title": self._generate_event_title(event_type, primary_country),
            "description": self._generate_event_description(event_type, primary_country, region),
            "region": region,
            "primary_country": primary_country,
            "severity": severity,
            "start_date": current_time.isoformat(),
            "end_date": end_date.isoformat(),
            "active": True
        }
        
        # 某些事件可能有影响的国家
        if event_type in ["diplomatic_crisis", "treaty_signing", "sanctions_imposed"]:
            # 选择另一个相关国家
            secondary_country = primary_country
            while secondary_country == primary_country:
                secondary_country = random_state.choice(countries)
            event["secondary_country"] = secondary_country
        
        self.logger.info(f"生成政治事件: {event['title']}")
        
        return event
    
    def _generate_event_title(self, event_type: str, country: str) -> str:
        """
        生成事件标题
        
        Args:
            event_type: 事件类型
            country: 主要国家
            
        Returns:
            事件标题
        """
        titles = {
            "election": f"{country} 举行全国大选",
            "protest": f"{country} 爆发大规模抗议活动",
            "coup_attempt": f"{country} 发生政变未遂",
            "diplomatic_crisis": f"与 {country} 的外交危机升级",
            "treaty_signing": f"{country} 签署重要国际条约",
            "leadership_change": f"{country} 领导层更迭",
            "sanctions_imposed": f"国际社会对 {country} 实施制裁",
            "terrorist_attack": f"{country} 遭遇恐怖袭击",
            "natural_disaster_response": f"{country} 应对重大自然灾害",
            "scandal_revelation": f"{country} 爆发重大政治丑闻"
        }
        
        return titles.get(event_type, f"{country} 发生政治事件")
    
    def _generate_event_description(self, event_type: str, country: str, region: str) -> str:
        """
        生成事件描述
        
        Args:
            event_type: 事件类型
            country: 主要国家
            region: 地区
            
        Returns:
            事件描述
        """
        descriptions = {
            "election": f"{country} 举行了备受关注的全国大选，结果将影响该国未来{random.randint(3, 5)}年的政治走向。",
            "protest": f"由于经济困难和政治不满，{country} 爆发了大规模抗议活动，数万人走上街头表达诉求。",
            "coup_attempt": f"{country} 军方部分派系试图推翻现政府，但政变行动被迅速镇压。",
            "diplomatic_crisis": f"因领土争端和历史问题，与 {country} 的外交关系急剧恶化，双方互相驱逐外交官。",
            "treaty_signing": f"{country} 与多国在{region}签署了历史性条约，旨在促进地区和平与经济合作。",
            "leadership_change": f"在党内权力斗争后，{country} 领导层发生重大变化，新领导人承诺推进改革。",
            "sanctions_imposed": f"联合国安理会通过决议，对 {country} 实施全面经济和外交制裁，理由是违反国际法。",
            "terrorist_attack": f"{country} 主要城市遭遇恐怖袭击，造成多人伤亡，政府宣布全国进入紧急状态。",
            "natural_disaster_response": f"面对{random.choice(['地震', '洪水', '飓风'])}等自然灾害，{country} 政府启动了全国性应急响应机制。",
            "scandal_revelation": f"媒体披露了 {country} 高级官员涉嫌腐败的证据，引发公众强烈不满和要求调查的呼声。"
        }
        
        return descriptions.get(event_type, f"{country} 发生了重大政治事件，详情尚未完全披露。")
    
    def calculate_political_stability_index(self, political_state: Dict[str, Any]) -> Dict[str, float]:
        """
        计算政治稳定性指数
        
        Args:
            political_state: 政治状态
            
        Returns:
            各国家/地区的政治稳定性指数（0-100）
        """
        stability_indices = {}
        
        # 计算地区政治稳定性
        if "tensions" in political_state:
            for region, tension in political_state["tensions"].items():
                # 紧张度越低，稳定性越高
                stability = 100 - tension
                stability_indices[region] = stability
        
        # 计算国家政治稳定性
        if "regimes" in political_state:
            for country, regime_info in political_state["regimes"].items():
                stability = regime_info.get("stability", 50)
                stability_indices[country] = stability
        
        # 综合评估
        global_stability = 50.0
        if stability_indices:
            global_stability = sum(stability_indices.values()) / len(stability_indices)
        stability_indices["global"] = global_stability
        
        return stability_indices

# 测试代码
if __name__ == "__main__":
    # 简单的测试配置
    config = {
        "political_volatility": 0.05,
        "alliance_stability": 0.8,
        "conflict_decay_rate": 0.1,
        "election_cycle": 4,
        "country_parameters": {
            "US": {
                "election_cycle": 4,
                "political_parties": ["Democratic Party", "Republican Party", "Libertarian Party"]
            },
            "China": {
                "election_cycle": 5,
                "political_parties": ["Communist Party of China"]
            },
            "EU": {
                "election_cycle": 5,
                "political_parties": ["EPP", "S&D", "Renew Europe", "ECR", "GUE/NGL", "Greens/EFA"]
            }
        }
    }
    
    # 创建政治模型
    pol_model = PoliticalModel(config)
    
    # 创建初始政治状态
    initial_state = {
        "tensions": {
            "Middle East": 85.0,
            "East Asia": 60.0,
            "Europe": 40.0,
            "South Asia": 70.0,
            "Americas": 30.0
        },
        "alliances": [
            "NATO",
            "EU",
            "ASEAN",
            "BRICS",
            "G7"
        ],
        "diplomacy": {
            "US": {
                "US": 100.0,
                "China": -30.0,
                "Russia": -60.0,
                "EU": 70.0,
                "India": 50.0
            },
            "China": {
                "US": -35.0,
                "China": 100.0,
                "Russia": 60.0,
                "EU": 40.0,
                "India": -20.0
            },
            "EU": {
                "US": 65.0,
                "China": 35.0,
                "Russia": -55.0,
                "EU": 100.0,
                "India": 55.0
            }
        },
        "regimes": {
            "US": {
                "type": "democracy",
                "government": "Democratic Party",
                "stability": 65.0,
                "last_election": (datetime.now() - timedelta(days=730)).isoformat(),
                "established_date": (datetime.now() - timedelta(days=730)).isoformat()
            },
            "China": {
                "type": "socialist",
                "government": "Communist Party of China",
                "stability": 85.0,
                "last_election": (datetime.now() - timedelta(days=1095)).isoformat(),
                "established_date": (datetime.now() - timedelta(days=1095)).isoformat()
            },
            "EU": {
                "type": "supranational",
                "government": "EPP Coalition",
                "stability": 70.0,
                "last_election": (datetime.now() - timedelta(days=1460)).isoformat(),
                "established_date": (datetime.now() - timedelta(days=1460)).isoformat()
            }
        },
        "events": [
            {
                "id": "event_1",
                "type": "diplomatic_crisis",
                "title": "与 Russia 的外交危机",
                "description": "因乌克兰冲突，与 Russia 的关系急剧恶化。",
                "region": "Europe",
                "primary_country": "EU",
                "secondary_country": "Russia",
                "severity": 4,
                "start_date": (datetime.now() - timedelta(days=300)).isoformat(),
                "end_date": (datetime.now() + timedelta(days=365)).isoformat(),
                "active": True
            }
        ]
    }
    
    # 模拟演化
    current_time = datetime.now()
    random_state = np.random.RandomState(42)
    
    print("初始政治状态:")
    print(f"地区紧张度: {initial_state['tensions']}")
    print(f"主要联盟: {initial_state['alliances']}")
    print(f"美国外交关系: {initial_state['diplomacy']['US']}")
    print()
    
    # 演化10步
    for step in range(10):
        next_time = current_time + timedelta(days=30)  # 假设每月一步
        new_state = pol_model.evolve(initial_state, next_time, random_state)
        initial_state = new_state
    
    print("演化后政治状态:")
    print(f"地区紧张度: {initial_state['tensions']}")
    print(f"主要联盟: {initial_state['alliances']}")
    print(f"美国外交关系: {initial_state['diplomacy']['US']}")
    
    # 如果有新事件
    if len(initial_state['events']) > 1:
        print(f"\n新增事件:")
        for event in initial_state['events']:
            if event['id'] != "event_1":
                print(f"- {event['title']} (类型: {event['type']}, 严重度: {event['severity']})")
    
    # 计算政治稳定性指数
    stability_indices = pol_model.calculate_political_stability_index(initial_state)
    print(f"\n政治稳定性指数:")
    for entity, index in stability_indices.items():
        print(f"{entity}: {index:.2f}")