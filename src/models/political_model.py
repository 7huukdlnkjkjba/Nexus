#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
政治模型模块
负责模拟全球政治格局的变化
"""

import logging
import random
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import math

from ..utils.logger import get_logger
from ..utils.random_utils import get_seeded_random


class PoliticalModel:
    """
    政治模型
    模拟全球政治格局的变化
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化政治模型
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("PoliticalModel")
        self.config = config
        
        # 政治参数配置
        self.stability_params = config.get("stability", {
            "volatility": 0.05,        # 稳定性波动范围
            "event_impact": 0.15,      # 事件对稳定性的影响程度
            "economic_sensitivity": 0.3 # 经济对稳定性的影响敏感度
        })
        
        self.relations_params = config.get("relations", {
            "volatility": 0.08,        # 关系波动范围
            "alliance_impact": 0.25,   # 联盟对关系的影响
            "ideology_impact": 0.2,    # 意识形态对关系的影响
            "economic_impact": 0.15    # 经济依赖对关系的影响
        })
        
        # 国家政治属性
        self.political_powers = config.get("political_powers", [
            "US", "China", "Russia", "EU", "UK", "France", "Germany",
            "India", "Japan", "Brazil", "Australia", "Canada", "South_Korea",
            "Turkey", "Saudi_Arabia", "Israel", "Iran", "Pakistan", "North_Korea"
        ])
        
        # 初始政治意识形态分类
        self.ideologies = config.get("ideologies", {
            "liberal_democracy": ["US", "EU", "UK", "France", "Germany", 
                                  "Japan", "Canada", "Australia", "South_Korea"],
            "authoritarian": ["China", "Russia", "North_Korea", "Iran", 
                              "Saudi_Arabia"],
            "hybrid": ["India", "Brazil", "Turkey", "Pakistan"],
            "theocracy": ["Iran", "Saudi_Arabia"],
            "military": ["North_Korea", "Pakistan"]
        })
        
        # 初始联盟关系
        self.alliances = config.get("alliances", {
            "NATO": ["US", "Canada", "UK", "France", "Germany", 
                     "Turkey", "Italy", "Spain", "Netherlands", 
                     "Belgium", "Luxembourg", "Denmark", "Norway", 
                     "Iceland", "Portugal", "Greece", "Poland"],
            "EU": ["Germany", "France", "Italy", "Spain", "Netherlands", 
                   "Belgium", "Poland", "Romania", "Sweden"],
            "SCO": ["China", "Russia", "India", "Pakistan", "Uzbekistan", 
                    "Kazakhstan", "Kyrgyzstan", "Tajikistan"],
            "ASEAN": ["Indonesia", "Thailand", "Vietnam", "Singapore", 
                      "Philippines", "Malaysia", "Myanmar", "Cambodia"],
            "Five_Eyes": ["US", "UK", "Canada", "Australia", "New_Zealand"]
        })
        
        # 政治事件概率配置
        self.event_probabilities = config.get("event_probabilities", {
            "election": 0.05,           # 选举概率
            "crisis": 0.03,             # 政治危机概率
            "regime_change": 0.01,      # 政权更迭概率
            "conflict": 0.02,           # 冲突爆发概率
            "alliance_change": 0.015    # 联盟变更概率
        })
        
        self.logger.info("政治模型初始化完成")
    
    def initialize_state(self, initial_year: int) -> Dict[str, Any]:
        """
        初始化政治状态
        
        Args:
            initial_year: 初始年份
            
        Returns:
            初始化后的政治状态字典
        """
        # 创建基础政治状态
        political_state = {
            "year": initial_year,
            "stability": {},
            "relations": {},
            "governments": {},
            "alliances": self.alliances.copy(),
            "ideologies": self.ideologies.copy()
        }
        
        # 为主要政治力量初始化稳定性值
        for country in self.political_powers:
            # 初始化稳定性值（0-1之间）
            if country in ["US", "UK", "Germany", "France", "Japan", "Canada"]:
                political_state["stability"][country] = 0.85
            elif country in ["China", "Russia", "Saudi_Arabia"]:
                political_state["stability"][country] = 0.80
            elif country in ["India", "Brazil", "South_Korea"]:
                political_state["stability"][country] = 0.75
            else:
                # 其他国家的基础稳定性值
                political_state["stability"][country] = 0.70
            
            # 初始化政府类型
            if country in self.ideologies["liberal_democracy"]:
                political_state["governments"][country] = "liberal_democracy"
            elif country in self.ideologies["authoritarian"]:
                political_state["governments"][country] = "authoritarian"
            elif country in self.ideologies["hybrid"]:
                political_state["governments"][country] = "hybrid"
            elif country in self.ideologies["theocracy"]:
                political_state["governments"][country] = "theocracy"
            elif country in self.ideologies["military"]:
                political_state["governments"][country] = "military"
            else:
                political_state["governments"][country] = "unknown"
        
        # 初始化国家间关系
        for source in self.political_powers:
            political_state["relations"][source] = {}
            for target in self.political_powers:
                if source == target:
                    # 国家与自身的关系为1.0
                    political_state["relations"][source][target] = 1.0
                elif self._are_allies(source, target):
                    # 盟国关系较好
                    political_state["relations"][source][target] = 0.7
                elif self._same_ideology(source, target):
                    # 相同意识形态关系一般较好
                    political_state["relations"][source][target] = 0.6
                else:
                    # 其他情况的基础关系值
                    political_state["relations"][source][target] = 0.4
        
        return political_state
    
    def _are_allies(self, country1: str, country2: str) -> bool:
        """
        检查两个国家是否为盟国
        
        Args:
            country1: 第一个国家
            country2: 第二个国家
            
        Returns:
            是否为盟国
        """
        for alliance, members in self.alliances.items():
            if country1 in members and country2 in members:
                return True
        return False
    
    def _same_ideology(self, country1: str, country2: str) -> bool:
        """
        检查两个国家是否有相同的主要意识形态
        
        Args:
            country1: 第一个国家
            country2: 第二个国家
            
        Returns:
            是否有相同意识形态
        """
        for ideology, countries in self.ideologies.items():
            if country1 in countries and country2 in countries:
                return True
        return False
    
    def evolve(self, political_state: Dict[str, Any], current_time: datetime, 
              economic_state: Optional[Dict[str, Any]] = None,
              rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """
        演化政治状态
        
        Args:
            political_state: 当前政治状态
            current_time: 当前时间
            economic_state: 经济状态（可选）
            rng: 随机数生成器
            
        Returns:
            演化后的政治状态
        """
        try:
            # 创建新状态以避免修改原始数据
            new_state = political_state.copy()
            
            # 确保所有必要的子状态存在
            new_state["stability"] = new_state.get("stability", {})
            new_state["relations"] = new_state.get("relations", {})
            new_state["events"] = new_state.get("events", [])
            new_state["alliances"] = new_state.get("alliances", {})
            new_state["leadership"] = new_state.get("leadership", {})
            
            # 初始化随机数生成器
            if rng is None:
                rng = get_seeded_random(hash(str(current_time)))
            
            # 生成随机政治事件
            new_events = self._generate_political_events(
                new_state, current_time, economic_state, rng
            )
            new_state["events"].extend(new_events)
            
            # 应用政治事件的影响
            self._apply_event_effects(new_state, new_events, rng)
            
            # 更新国家稳定性
            new_state["stability"] = self._update_stability(
                new_state["stability"], new_state, economic_state, rng
            )
            
            # 更新国家间关系
            new_state["relations"] = self._update_relations(
                new_state["relations"], new_state, economic_state, rng
            )
            
            # 更新联盟状态
            new_state["alliances"] = self._update_alliances(new_state, rng)
            
            # 更新领导层（如果有选举）
            new_state["leadership"] = self._update_leadership(new_state, rng)
            
            return new_state
            
        except Exception as e:
            self.logger.error(f"演化政治状态失败: {str(e)}")
            return political_state  # 失败时返回原始状态
    
    def _generate_political_events(self, political_state: Dict[str, Any], 
                                  current_time: datetime, 
                                  economic_state: Optional[Dict[str, Any]] = None,
                                  rng: random.Random = None) -> List[Dict[str, Any]]:
        """
        生成随机政治事件
        
        Args:
            political_state: 政治状态
            current_time: 当前时间
            economic_state: 经济状态（可选）
            rng: 随机数生成器
            
        Returns:
            生成的事件列表
        """
        events = []
        year = current_time.year
        
        # 检查各种事件的发生概率
        for event_type, probability in self.event_probabilities.items():
            if rng.random() < probability:
                event = self._create_event(event_type, political_state, 
                                         economic_state, year, rng)
                if event:
                    events.append(event)
        
        # 基于经济状态触发额外事件
        if economic_state and rng.random() < 0.02:
            event = self._create_economic_related_event(economic_state, year, rng)
            if event:
                events.append(event)
        
        # 基于不稳定国家触发冲突
        if rng.random() < 0.01:
            event = self._create_conflict_event(political_state, year, rng)
            if event:
                events.append(event)
        
        return events
    
    def _create_event(self, event_type: str, political_state: Dict[str, Any],
                     economic_state: Optional[Dict[str, Any]], 
                     year: int, rng: random.Random) -> Optional[Dict[str, Any]]:
        """
        创建特定类型的政治事件
        
        Args:
            event_type: 事件类型
            political_state: 政治状态
            economic_state: 经济状态（可选）
            year: 年份
            rng: 随机数生成器
            
        Returns:
            事件字典
        """
        countries = self.political_powers
        
        if event_type == "election":
            country = rng.choice(countries)
            # 判断是否是民主国家（简化判断）
            is_democracy = any(country in alliance for alliance in ["NATO", "EU", "Five_Eyes"])
            if is_democracy:
                return {
                    "type": "election",
                    "country": country,
                    "year": year,
                    "impact": rng.uniform(-0.1, 0.1),
                    "description": f"{country} 举行全国大选"
                }
        
        elif event_type == "crisis":
            country = rng.choice(countries)
            return {
                "type": "crisis",
                "country": country,
                "year": year,
                "impact": rng.uniform(-0.2, -0.05),
                "description": f"{country} 面临政治危机"
            }
        
        elif event_type == "regime_change":
            # 更容易在不稳定国家发生
            stability = political_state.get("stability", {})
            unstable_countries = [c for c in countries 
                                if stability.get(c, 0.5) < 0.4]
            
            if unstable_countries:
                country = rng.choice(unstable_countries)
            else:
                country = rng.choice(countries)
                
            return {
                "type": "regime_change",
                "country": country,
                "year": year,
                "impact": rng.uniform(-0.3, 0.1),
                "description": f"{country} 发生政权更迭"
            }
        
        elif event_type == "conflict":
            # 选择两个关系不好的国家
            relations = political_state.get("relations", {})
            potential_conflicts = []
            
            for country1 in countries:
                for country2 in countries:
                    if country1 != country2:
                        rel_value = relations.get(f"{country1}-{country2}", 0)
                        if rel_value < 0:
                            potential_conflicts.append((country1, country2, rel_value))
            
            if potential_conflicts:
                # 按关系最差排序
                potential_conflicts.sort(key=lambda x: x[2])
                # 选择前几个最差关系之一
                country1, country2, _ = rng.choice(potential_conflicts[:min(5, len(potential_conflicts))])
                
                return {
                    "type": "conflict",
                    "countries": [country1, country2],
                    "year": year,
                    "impact": rng.uniform(-0.3, -0.1),
                    "description": f"{country1} 与 {country2} 之间爆发冲突"
                }
        
        elif event_type == "alliance_change":
            # 联盟变更
            alliance_names = list(self.alliances.keys())
            if alliance_names:
                alliance = rng.choice(alliance_names)
                action = rng.choice(["expansion", "withdrawal"])
                country_pool = []
                
                if action == "expansion":
                    # 选择不在联盟中的国家
                    alliance_countries = set(self.alliances.get(alliance, []))
                    country_pool = [c for c in countries if c not in alliance_countries]
                else:
                    # 选择联盟中的国家
                    country_pool = self.alliances.get(alliance, [])
                
                if country_pool:
                    country = rng.choice(country_pool)
                    
                    return {
                        "type": "alliance_change",
                        "alliance": alliance,
                        "action": action,
                        "country": country,
                        "year": year,
                        "impact": rng.uniform(-0.1, 0.1),
                        "description": f"{country} {'加入' if action == 'expansion' else '退出'} {alliance}"
                    }
        
        return None
    
    def _create_economic_related_event(self, economic_state: Dict[str, Any], 
                                     year: int, rng: random.Random) -> Optional[Dict[str, Any]]:
        """
        创建与经济相关的政治事件
        
        Args:
            economic_state: 经济状态
            year: 年份
            rng: 随机数生成器
            
        Returns:
            事件字典
        """
        gdp_data = economic_state.get("gdp", {})
        inflation_data = economic_state.get("inflation", {})
        unemployment_data = economic_state.get("unemployment", {})
        
        # 找出经济状况最差的国家
        struggling_countries = []
        
        for country in self.political_powers:
            score = 0
            if country in gdp_data:
                # 简化评分：通胀高和失业率高的国家经济状况差
                if country in inflation_data and inflation_data[country] > 10:
                    score -= 1
                if country in unemployment_data and unemployment_data[country] > 15:
                    score -= 1
                
                if score < 0:
                    struggling_countries.append(country)
        
        if struggling_countries:
            country = rng.choice(struggling_countries)
            event_type = rng.choice(["protest", "policy_shift", "austerity_measures"])
            
            descriptions = {
                "protest": f"{country} 因经济困难爆发大规模抗议",
                "policy_shift": f"{country} 实施经济政策重大转向",
                "austerity_measures": f"{country} 推行紧缩政策应对经济危机"
            }
            
            return {
                "type": "economic_political",
                "subtype": event_type,
                "country": country,
                "year": year,
                "impact": rng.uniform(-0.2, -0.05),
                "description": descriptions[event_type]
            }
        
        return None
    
    def _create_conflict_event(self, political_state: Dict[str, Any], 
                             year: int, rng: random.Random) -> Optional[Dict[str, Any]]:
        """
        创建冲突事件
        
        Args:
            political_state: 政治状态
            year: 年份
            rng: 随机数生成器
            
        Returns:
            事件字典
        """
        stability = political_state.get("stability", {})
        
        # 找出最不稳定的国家
        unstable_countries = sorted(
            [(c, stability.get(c, 0.5)) for c in self.political_powers],
            key=lambda x: x[1]
        )
        
        if unstable_countries:
            unstable_country, _ = unstable_countries[0]  # 最不稳定的国家
            
            # 随机选择一个对手（可能是邻国或竞争对手）
            potential_rivals = [c for c in self.political_powers if c != unstable_country]
            
            if potential_rivals:
                rival = rng.choice(potential_rivals)
                conflict_type = rng.choice(["border_conflict", "proxy_conflict", "diplomatic_crisis"])
                
                descriptions = {
                    "border_conflict": f"{unstable_country} 与 {rival} 发生边境冲突",
                    "proxy_conflict": f"{unstable_country} 与 {rival} 支持第三方冲突",
                    "diplomatic_crisis": f"{unstable_country} 与 {rival} 爆发外交危机"
                }
                
                return {
                    "type": "conflict",
                    "subtype": conflict_type,
                    "countries": [unstable_country, rival],
                    "year": year,
                    "impact": rng.uniform(-0.25, -0.08),
                    "description": descriptions[conflict_type]
                }
        
        return None
    
    def _apply_event_effects(self, political_state: Dict[str, Any], 
                           events: List[Dict[str, Any]], rng: random.Random):
        """
        应用政治事件的影响
        
        Args:
            political_state: 政治状态
            events: 发生的事件
            rng: 随机数生成器
        """
        for event in events:
            event_type = event.get("type")
            
            if event_type == "election":
                country = event.get("country")
                impact = event.get("impact", 0)
                # 更新国家稳定性
                if country in political_state["stability"]:
                    political_state["stability"][country] = \
                        max(0, min(1, political_state["stability"][country] + impact))
            
            elif event_type == "crisis" or event_type == "regime_change":
                country = event.get("country")
                impact = event.get("impact", -0.1)
                # 降低国家稳定性
                if country in political_state["stability"]:
                    political_state["stability"][country] = \
                        max(0, min(1, political_state["stability"][country] + impact))
            
            elif event_type == "conflict":
                countries = event.get("countries", [])
                impact = event.get("impact", -0.1)
                # 降低冲突国家的稳定性和相互关系
                for country in countries:
                    if country in political_state["stability"]:
                        political_state["stability"][country] = \
                            max(0, min(1, political_state["stability"][country] + impact))
                
                # 恶化两国关系
                if len(countries) >= 2:
                    rel_key = f"{countries[0]}-{countries[1]}"
                    if rel_key in political_state["relations"]:
                        political_state["relations"][rel_key] = \
                            max(-1, min(1, political_state["relations"][rel_key] + impact * 0.5))
                    # 对称关系
                    rel_key2 = f"{countries[1]}-{countries[0]}"
                    if rel_key2 in political_state["relations"]:
                        political_state["relations"][rel_key2] = \
                            max(-1, min(1, political_state["relations"][rel_key2] + impact * 0.5))
            
            elif event_type == "alliance_change":
                alliance = event.get("alliance")
                country = event.get("country")
                action = event.get("action")
                
                # 更新联盟成员
                if alliance not in political_state["alliances"]:
                    political_state["alliances"][alliance] = self.alliances[alliance].copy()
                
                if action == "expansion" and country not in political_state["alliances"][alliance]:
                    political_state["alliances"][alliance].append(country)
                elif action == "withdrawal" and country in political_state["alliances"][alliance]:
                    political_state["alliances"][alliance].remove(country)
    
    def _update_stability(self, stability_data: Dict[str, float], 
                         political_state: Dict[str, Any],
                         economic_state: Optional[Dict[str, Any]], 
                         rng: random.Random) -> Dict[str, float]:
        """
        更新国家稳定性
        
        Args:
            stability_data: 当前稳定性数据
            political_state: 政治状态
            economic_state: 经济状态（可选）
            rng: 随机数生成器
            
        Returns:
            更新后的稳定性数据
        """
        new_stability = stability_data.copy()
        volatility = self.stability_params["volatility"]
        
        # 确保所有主要政治力量都有稳定性数据
        for country in self.political_powers:
            if country not in new_stability:
                # 设置默认稳定性（0-1范围）
                new_stability[country] = 0.7 + rng.uniform(-0.2, 0.2)
        
        # 更新每个国家的稳定性
        for country, current_stability in new_stability.items():
            # 基础随机波动
            random_change = rng.uniform(-volatility, volatility)
            
            # 经济因素影响
            economic_impact = 0
            if economic_state:
                economic_impact = self._calculate_economic_stability_impact(country, economic_state, rng)
            
            # 政治事件影响已经在_apply_event_effects中处理
            
            # 总变化
            total_change = random_change + economic_impact
            
            # 更新稳定性
            new_stability[country] = max(0, min(1, current_stability + total_change))
        
        return new_stability
    
    def _calculate_economic_stability_impact(self, country: str, 
                                           economic_state: Dict[str, Any],
                                           rng: random.Random) -> float:
        """
        计算经济对政治稳定性的影响
        
        Args:
            country: 国家名称
            economic_state: 经济状态
            rng: 随机数生成器
            
        Returns:
            经济影响值
        """
        impact = 0
        sensitivity = self.stability_params["economic_sensitivity"]
        
        # 检查GDP增长率（简化计算）
        gdp_data = economic_state.get("gdp", {})
        inflation_data = economic_state.get("inflation", {})
        unemployment_data = economic_state.get("unemployment", {})
        
        if country in gdp_data:
            # GDP增长对稳定性的影响
            gdp_change = rng.uniform(-0.03, 0.06)  # 模拟GDP增长率变化
            impact += gdp_change * sensitivity * 0.4
        
        if country in inflation_data:
            # 高通胀降低稳定性
            inflation = inflation_data[country]
            if inflation > 10:
                impact -= 0.1 * sensitivity
            elif inflation > 5:
                impact -= 0.05 * sensitivity
        
        if country in unemployment_data:
            # 高失业率降低稳定性
            unemployment = unemployment_data[country]
            if unemployment > 15:
                impact -= 0.15 * sensitivity
            elif unemployment > 10:
                impact -= 0.1 * sensitivity
            elif unemployment > 7:
                impact -= 0.05 * sensitivity
        
        return impact
    
    def _update_relations(self, relations_data: Dict[str, float], 
                         political_state: Dict[str, Any],
                         economic_state: Optional[Dict[str, Any]], 
                         rng: random.Random) -> Dict[str, float]:
        """
        更新国家间关系
        
        Args:
            relations_data: 当前关系数据
            political_state: 政治状态
            economic_state: 经济状态（可选）
            rng: 随机数生成器
            
        Returns:
            更新后的关系数据
        """
        new_relations = relations_data.copy()
        volatility = self.relations_params["volatility"]
        
        # 初始化所有国家对的关系
        for i, country1 in enumerate(self.political_powers):
            for country2 in self.political_powers[i+1:]:
                key1 = f"{country1}-{country2}"
                key2 = f"{country2}-{country1}"
                
                if key1 not in new_relations:
                    # 默认关系值（-1到1）
                    # 联盟成员关系较好，意识形态相似的关系较好
                    base_relation = 0
                    
                    # 检查是否是同一联盟
                    for alliance, members in self.alliances.items():
                        if country1 in members and country2 in members:
                            base_relation += 0.3
                            break
                    
                    # 检查意识形态相似度
                    base_relation += self._calculate_ideology_similarity(country1, country2)
                    
                    # 随机调整
                    base_relation += rng.uniform(-0.2, 0.2)
                    base_relation = max(-1, min(1, base_relation))
                    
                    new_relations[key1] = base_relation
                    new_relations[key2] = base_relation  # 关系对称
        
        # 更新每对国家的关系
        for key, current_relation in new_relations.items():
            country1, country2 = key.split("-")
            
            # 基础随机波动
            random_change = rng.uniform(-volatility, volatility)
            
            # 联盟影响
            alliance_impact = self._calculate_alliance_impact(country1, country2, political_state)
            
            # 意识形态影响
            ideology_impact = self._calculate_ideology_similarity(country1, country2) * 0.1
            
            # 经济依赖影响
            economic_impact = 0
            if economic_state:
                economic_impact = self._calculate_economic_relation_impact(
                    country1, country2, economic_state, rng
                )
            
            # 稳定性对关系的影响（不稳定国家更容易恶化关系）
            stability_impact = 0
            stability = political_state.get("stability", {})
            if country1 in stability and country2 in stability:
                avg_stability = (stability[country1] + stability[country2]) / 2
                if avg_stability < 0.4:
                    stability_impact = rng.uniform(-0.05, -0.01)
            
            # 总变化
            total_change = random_change + alliance_impact + ideology_impact + economic_impact + stability_impact
            
            # 更新关系
            new_relations[key] = max(-1, min(1, current_relation + total_change))
            
            # 确保关系对称性
            reverse_key = f"{country2}-{country1}"
            new_relations[reverse_key] = new_relations[key]
        
        return new_relations
    
    def _calculate_ideology_similarity(self, country1: str, country2: str) -> float:
        """
        计算两个国家的意识形态相似度
        
        Args:
            country1: 国家1
            country2: 国家2
            
        Returns:
            相似度得分（-1到1）
        """
        # 简化模型：共享相同意识形态类别的国家关系更好
        shared_categories = 0
        total_categories = 0
        
        for category, countries in self.ideologies.items():
            total_categories += 1
            if country1 in countries and country2 in countries:
                shared_categories += 1
        
        if total_categories == 0:
            return 0
        
        # 转换为-1到1的范围
        similarity = (2 * shared_categories / total_categories) - 1
        return similarity * self.relations_params["ideology_impact"]
    
    def _calculate_alliance_impact(self, country1: str, country2: str, 
                                 political_state: Dict[str, Any]) -> float:
        """
        计算联盟对国家关系的影响
        
        Args:
            country1: 国家1
            country2: 国家2
            political_state: 政治状态
            
        Returns:
            联盟影响值
        """
        alliances = political_state.get("alliances", {})
        shared_alliances = 0
        conflicting_alliances = 0
        
        # 检查共享的联盟
        for alliance, members in alliances.items():
            if country1 in members and country2 in members:
                shared_alliances += 1
        
        # 这里简化处理，实际应该考虑联盟之间的对抗关系
        impact = shared_alliances * self.relations_params["alliance_impact"] * 0.1
        return impact
    
    def _calculate_economic_relation_impact(self, country1: str, country2: str, 
                                          economic_state: Dict[str, Any],
                                          rng: random.Random) -> float:
        """
        计算经济依赖对国家关系的影响
        
        Args:
            country1: 国家1
            country2: 国家2
            economic_state: 经济状态
            rng: 随机数生成器
            
        Returns:
            经济影响值
        """
        trade_data = economic_state.get("trade", {})
        impact = 0
        
        # 检查两国之间的贸易关系
        if country1 in trade_data and country2 in trade_data[country1]:
            # 贸易额越大，关系越好
            trade_volume = trade_data[country1][country2]
            # 简化：贸易额对关系的影响
            impact += min(trade_volume / 100, 0.2) * self.relations_params["economic_impact"]
        
        return impact
    
    def _update_alliances(self, political_state: Dict[str, Any], 
                         rng: random.Random) -> Dict[str, List[str]]:
        """
        更新联盟状态
        
        Args:
            political_state: 政治状态
            rng: 随机数生成器
            
        Returns:
            更新后的联盟数据
        """
        alliances = political_state.get("alliances", {})
        new_alliances = {}
        
        # 复制现有联盟结构
        for alliance, members in alliances.items():
            new_alliances[alliance] = members.copy()
        
        # 如果没有联盟数据，初始化
        if not new_alliances:
            for alliance, members in self.alliances.items():
                new_alliances[alliance] = members.copy()
        
        return new_alliances
    
    def _update_leadership(self, political_state: Dict[str, Any], 
                          rng: random.Random) -> Dict[str, Any]:
        """
        更新领导层信息
        
        Args:
            political_state: 政治状态
            rng: 随机数生成器
            
        Returns:
            更新后的领导层数据
        """
        leadership = political_state.get("leadership", {})
        new_leadership = leadership.copy()
        
        # 检查是否有选举事件，可能导致领导层变更
        events = political_state.get("events", [])
        for event in events:
            if event.get("type") == "election":
                country = event.get("country")
                # 选举可能导致领导层变更
                if rng.random() < 0.4:  # 40%的概率领导层变更
                    ideology_shift = rng.choice([-0.1, 0, 0.1])  # 意识形态可能小幅变化
                    new_leadership[country] = {
                        "year": event.get("year"),
                        "ideology_shift": ideology_shift,
                        "description": f"{country} 新领导层上台，意识形态变化: {ideology_shift:.2f}"
                    }
        
        return new_leadership
    
    def calculate_political_risk(self, political_state: Dict[str, Any]) -> float:
        """
        计算整体政治风险
        
        Args:
            political_state: 政治状态
            
        Returns:
            0-1之间的风险分数（越高风险越大）
        """
        if not political_state:
            return 0.5
        
        stability = political_state.get("stability", {})
        relations = political_state.get("relations", {})
        events = political_state.get("events", [])
        
        risk_score = 0
        weights = {
            "stability": 0.4,
            "relations": 0.4,
            "events": 0.2
        }
        
        # 稳定性风险（稳定性越低，风险越高）
        if stability:
            avg_stability = sum(stability.values()) / len(stability)
            stability_risk = 1 - avg_stability  # 转换为风险分数
            risk_score += stability_risk * weights["stability"]
        
        # 关系风险（关系越差，风险越高）
        if relations:
            negative_relations = [r for r in relations.values() if r < 0]
            if negative_relations:
                avg_negative_relation = abs(sum(negative_relations) / len(negative_relations))
                relations_risk = avg_negative_relation
                risk_score += relations_risk * weights["relations"]
        
        # 事件风险
        negative_events = [e for e in events if e.get("impact", 0) < 0]
        if negative_events:
            event_risk = len(negative_events) / 10  # 每个负面事件增加0.1风险
            event_risk = min(event_risk, 1.0)  # 上限1.0
            risk_score += event_risk * weights["events"]
        
        return min(1.0, risk_score)  # 确保在0-1范围内


# 测试代码
if __name__ == "__main__":
    # 简单的测试配置
    config = {
        "stability": {
            "volatility": 0.05,
            "event_impact": 0.15
        },
        "relations": {
            "volatility": 0.08
        }
    }
    
    # 创建政治模型
    model = PoliticalModel(config)
    
    # 初始政治状态
    initial_state = {
        "stability": {
            "US": 0.75,
            "China": 0.8,
            "Russia": 0.6,
            "EU": 0.7,
            "UK": 0.65,
            "France": 0.7,
            "Germany": 0.72,
            "India": 0.68,
            "Japan": 0.85,
            "Brazil": 0.55
        },
        "relations": {
            "US-China": 0.2,
            "US-Russia": -0.6,
            "China-Russia": 0.5,
            "US-EU": 0.7,
            "China-EU": 0.3,
            "Russia-EU": -0.4
        },
        "events": [],
        "alliances": {},
        "leadership": {}
    }
    
    # 模拟演化
    from datetime import datetime
    
    print("初始政治状态:")
    print(f"稳定性: {initial_state['stability']}")
    print(f"关系: {initial_state['relations']}")
    
    # 演化一次
    new_state = model.evolve(initial_state, datetime.now())
    
    print("\n演化后的政治状态:")
    print(f"稳定性: {new_state['stability']}")
    print(f"关系: {new_state['relations']}")
    print(f"事件: {new_state['events']}")
    
    # 计算政治风险
    risk_score = model.calculate_political_risk(new_state)
    print(f"\n政治风险指数: {risk_score:.4f}")