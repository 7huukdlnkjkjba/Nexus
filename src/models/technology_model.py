#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
技术模型模块
负责模拟全球技术发展和创新
"""

import logging
import random
from typing import Dict, Any, List, Tuple
from datetime import datetime
import math

from ..utils.logger import get_logger
from ..utils.random_utils import get_seeded_random


class TechnologyModel:
    """
    技术模型
    模拟全球技术发展和创新
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化技术模型
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("TechnologyModel")
        self.config = config
        
        # 技术参数配置
        self.research_params = config.get("research", {
            "baseline_funding_growth": 0.03,    # 研发投入基准增长率
            "volatility": 0.02,                 # 波动范围
            "economic_impact": 0.4              # 经济对研发的影响因子
        })
        
        self.innovation_params = config.get("innovation", {
            "breakthrough_probability": 0.02,   # 重大突破概率
            "incremental_probability": 0.15,    # 渐进式创新概率
            "diffusion_rate": 0.08,             # 技术扩散率
            "obsolescence_rate": 0.03           # 技术淘汰率
        })
        
        # 主要技术领域
        self.technology_fields = config.get("technology_fields", [
            "artificial_intelligence",
            "quantum_computing",
            "biotechnology",
            "renewable_energy",
            "nanotechnology",
            "advanced_materials",
            "space_exploration",
            "robotics",
            "blockchain",
            "cybersecurity",
            "fusion_energy",
            "gene_editing",
            "brain_computer_interfaces",
            "autonomous_systems",
            "digital_twins"
        ])
        
        # 技术领域分类
        self.technology_categories = config.get("technology_categories", {
            "information_technology": ["artificial_intelligence", "quantum_computing", 
                                      "blockchain", "cybersecurity", "digital_twins"],
            "energy": ["renewable_energy", "fusion_energy"],
            "biotechnology": ["biotechnology", "gene_editing", "brain_computer_interfaces"],
            "advanced_materials": ["nanotechnology", "advanced_materials"],
            "automation": ["robotics", "autonomous_systems"],
            "space": ["space_exploration"]
        })
        
        # 技术领先国家
        self.tech_leaders = config.get("tech_leaders", {
            "US": ["artificial_intelligence", "quantum_computing", "biotechnology", 
                    "space_exploration", "robotics", "cybersecurity"],
            "China": ["artificial_intelligence", "renewable_energy", "robotics", 
                     "space_exploration", "quantum_computing"],
            "EU": ["renewable_energy", "biotechnology", "advanced_materials"],
            "Japan": ["robotics", "advanced_materials", "autonomous_systems"],
            "South_Korea": ["semiconductors", "robotics", "advanced_materials"],
            "Israel": ["cybersecurity", "biotechnology"],
            "Germany": ["advanced_materials", "automation", "renewable_energy"],
            "UK": ["artificial_intelligence", "quantum_computing", "biotechnology"]
        })
        
        # 技术成熟度曲线配置（基于Gartner曲线）
        self.maturity_phases = {
            "innovation_trigger": 0.1,
            "peak_of_inflated_expectations": 0.3,
            "trough_of_disillusionment": 0.4,
            "slope_of_enlightenment": 0.6,
            "plateau_of_productivity": 0.8
        }
        
        self.logger.info("技术模型初始化完成")
    
    def evolve(self, technology_state: Dict[str, Any], current_time: datetime, 
              economic_state: Optional[Dict[str, Any]] = None,
              rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """
        演化技术状态
        
        Args:
            technology_state: 当前技术状态
            current_time: 当前时间
            economic_state: 经济状态（可选）
            rng: 随机数生成器
            
        Returns:
            演化后的技术状态
        """
        try:
            # 创建新状态以避免修改原始数据
            new_state = technology_state.copy()
            
            # 确保所有必要的子状态存在
            new_state["technologies"] = new_state.get("technologies", {})
            new_state["research_funding"] = new_state.get("research_funding", {})
            new_state["breakthroughs"] = new_state.get("breakthroughs", [])
            new_state["country_capabilities"] = new_state.get("country_capabilities", {})
            new_state["technology_adoption"] = new_state.get("technology_adoption", {})
            
            # 初始化随机数生成器
            if rng is None:
                rng = get_seeded_random(hash(str(current_time)))
            
            # 初始化技术状态（如果不存在）
            self._initialize_technologies(new_state)
            
            # 更新研究投入
            new_state["research_funding"] = self._update_research_funding(
                new_state["research_funding"], economic_state, rng
            )
            
            # 生成技术突破
            new_breakthroughs = self._generate_breakthroughs(
                new_state, new_state["research_funding"], current_time, rng
            )
            new_state["breakthroughs"].extend(new_breakthroughs)
            
            # 应用技术突破
            self._apply_breakthroughs(new_state, new_breakthroughs)
            
            # 更新技术成熟度
            new_state["technologies"] = self._update_technology_maturity(
                new_state["technologies"], new_state["research_funding"], rng
            )
            
            # 更新国家技术能力
            new_state["country_capabilities"] = self._update_country_capabilities(
                new_state["country_capabilities"], new_state["research_funding"], 
                new_state["technologies"], rng
            )
            
            # 更新技术采用率
            new_state["technology_adoption"] = self._update_technology_adoption(
                new_state["technology_adoption"], new_state["technologies"], 
                new_state["country_capabilities"], rng
            )
            
            return new_state
            
        except Exception as e:
            self.logger.error(f"演化技术状态失败: {str(e)}")
            return technology_state  # 失败时返回原始状态
    
    def _initialize_technologies(self, technology_state: Dict[str, Any]):
        """
        初始化技术状态
        
        Args:
            technology_state: 技术状态
        """
        technologies = technology_state["technologies"]
        
        # 为每个技术领域初始化状态
        for tech_field in self.technology_fields:
            if tech_field not in technologies:
                # 设置初始成熟度（0-1范围）
                initial_maturity = self._get_initial_maturity(tech_field)
                
                # 设置初始潜力（0-1范围）
                initial_potential = self._get_initial_potential(tech_field)
                
                technologies[tech_field] = {
                    "maturity": initial_maturity,
                    "potential": initial_potential,
                    "growth_rate": 0.05,  # 初始增长率
                    "investment_level": 0.03,  # 初始投资水平（GDP比例）
                    "key_countries": self._get_key_countries(tech_field),
                    "last_breakthrough": None,
                    "adoption_challenges": random.random() * 0.5  # 0-0.5的采用挑战
                }
    
    def _get_initial_maturity(self, tech_field: str) -> float:
        """
        获取技术领域的初始成熟度
        
        Args:
            tech_field: 技术领域
            
        Returns:
            初始成熟度（0-1）
        """
        # 基于当前技术发展状况的简化估计
        maturity_map = {
            "artificial_intelligence": 0.65,  # 处于斜坡期
            "quantum_computing": 0.25,         # 处于膨胀期望峰值
            "biotechnology": 0.6,              # 接近生产力平台期
            "renewable_energy": 0.75,          # 生产力平台期早期
            "nanotechnology": 0.45,            # 处于幻觉低谷
            "advanced_materials": 0.55,        # 开始爬坡期
            "space_exploration": 0.5,          # 幻觉低谷到爬坡期
            "robotics": 0.6,                   # 爬坡期
            "blockchain": 0.4,                 # 幻觉低谷
            "cybersecurity": 0.7,              # 生产力平台期
            "fusion_energy": 0.2,              # 膨胀期望峰值
            "gene_editing": 0.45,              # 幻觉低谷
            "brain_computer_interfaces": 0.3,  # 膨胀期望峰值
            "autonomous_systems": 0.5,         # 幻觉低谷
            "digital_twins": 0.4               # 幻觉低谷
        }
        
        return maturity_map.get(tech_field, 0.3) + random.uniform(-0.05, 0.05)
    
    def _get_initial_potential(self, tech_field: str) -> float:
        """
        获取技术领域的初始潜力
        
        Args:
            tech_field: 技术领域
            
        Returns:
            初始潜力（0-1）
        """
        # 基于技术变革潜力的估计
        potential_map = {
            "artificial_intelligence": 0.95,   # 极高潜力
            "quantum_computing": 0.9,          # 极高潜力
            "biotechnology": 0.85,             # 高潜力
            "renewable_energy": 0.8,           # 高潜力
            "nanotechnology": 0.85,            # 高潜力
            "advanced_materials": 0.75,        # 中高潜力
            "space_exploration": 0.7,          # 中高潜力
            "robotics": 0.8,                   # 高潜力
            "blockchain": 0.7,                 # 中高潜力
            "cybersecurity": 0.65,             # 中等潜力
            "fusion_energy": 0.95,             # 极高潜力
            "gene_editing": 0.9,               # 极高潜力
            "brain_computer_interfaces": 0.9,  # 极高潜力
            "autonomous_systems": 0.85,        # 高潜力
            "digital_twins": 0.75              # 中高潜力
        }
        
        return potential_map.get(tech_field, 0.7)
    
    def _get_key_countries(self, tech_field: str) -> List[str]:
        """
        获取在特定技术领域领先的国家
        
        Args:
            tech_field: 技术领域
            
        Returns:
            领先国家列表
        """
        key_countries = []
        for country, techs in self.tech_leaders.items():
            if tech_field in techs:
                key_countries.append(country)
        return key_countries
    
    def _update_research_funding(self, funding_data: Dict[str, float],
                               economic_state: Optional[Dict[str, Any]],
                               rng: random.Random) -> Dict[str, float]:
        """
        更新研究投入
        
        Args:
            funding_data: 当前研究投入数据（GDP百分比）
            economic_state: 经济状态（可选）
            rng: 随机数生成器
            
        Returns:
            更新后的研究投入数据
        """
        new_funding = funding_data.copy()
        baseline_growth = self.research_params["baseline_funding_growth"]
        volatility = self.research_params["volatility"]
        
        # 确保所有主要技术国家都有研究投入数据
        for country in self.tech_leaders.keys():
            if country not in new_funding:
                # 设置默认研究投入（GDP百分比）
                default_funding = self._get_default_research_funding(country)
                new_funding[country] = default_funding
        
        # 更新每个国家的研究投入
        for country, current_funding in new_funding.items():
            # 基础增长率
            growth = baseline_growth + rng.uniform(-volatility, volatility)
            
            # 经济因素影响
            if economic_state:
                economic_factor = self._calculate_economic_funding_impact(country, economic_state, rng)
                growth += economic_factor
            
            # 更新研究投入
            new_funding[country] = max(0.5, min(5.0, current_funding * (1 + growth)))  # 限制在0.5%-5%
        
        return new_funding
    
    def _get_default_research_funding(self, country: str) -> float:
        """
        获取国家默认研究投入比例
        
        Args:
            country: 国家名称
            
        Returns:
            默认研究投入（GDP百分比）
        """
        default_funding = {
            "US": 3.4,
            "China": 2.4,
            "EU": 2.3,
            "Japan": 3.2,
            "South_Korea": 4.8,
            "Israel": 5.4,
            "Germany": 3.1,
            "UK": 2.2,
            "France": 2.2,
            "Switzerland": 3.1,
            "Singapore": 3.3,
            "Sweden": 3.4
        }
        
        return default_funding.get(country, 1.5)  # 默认1.5%
    
    def _calculate_economic_funding_impact(self, country: str, 
                                         economic_state: Dict[str, Any],
                                         rng: random.Random) -> float:
        """
        计算经济对研究投入的影响
        
        Args:
            country: 国家名称
            economic_state: 经济状态
            rng: 随机数生成器
            
        Returns:
            经济影响值
        """
        impact = 0
        sensitivity = self.research_params["economic_impact"]
        
        # 检查GDP、通胀和失业率（简化模型）
        gdp_data = economic_state.get("gdp", {})
        inflation_data = economic_state.get("inflation", {})
        unemployment_data = economic_state.get("unemployment", {})
        
        # GDP增长影响
        if country in gdp_data:
            # 简化：假设GDP增长与研究投入正相关
            gdp_growth = rng.uniform(-0.02, 0.06)  # 模拟GDP增长率
            impact += gdp_growth * sensitivity * 0.3
        
        # 通胀影响（高通胀可能导致研究投入下降）
        if country in inflation_data and inflation_data[country] > 10:
            impact -= 0.01 * sensitivity
        
        # 失业率影响（高失业率可能导致研究投入下降）
        if country in unemployment_data and unemployment_data[country] > 15:
            impact -= 0.015 * sensitivity
        
        return impact
    
    def _generate_breakthroughs(self, technology_state: Dict[str, Any],
                              funding_data: Dict[str, float],
                              current_time: datetime,
                              rng: random.Random) -> List[Dict[str, Any]]:
        """
        生成技术突破
        
        Args:
            technology_state: 技术状态
            funding_data: 研究投入数据
            current_time: 当前时间
            rng: 随机数生成器
            
        Returns:
            生成的技术突破列表
        """
        breakthroughs = []
        technologies = technology_state["technologies"]
        
        # 为每个技术领域检查是否有突破
        for tech_field, tech_data in technologies.items():
            # 计算突破概率
            breakthrough_prob = self._calculate_breakthrough_probability(
                tech_field, tech_data, funding_data, rng
            )
            
            if rng.random() < breakthrough_prob:
                # 生成突破
                breakthrough_type = rng.choice(["major", "significant", "incremental"])
                impact = self._get_breakthrough_impact(breakthrough_type, tech_data)
                
                # 确定突破国家
                leading_countries = self._get_leading_countries(tech_field, funding_data)
                breakthrough_country = rng.choice(leading_countries) if leading_countries else "Global"
                
                breakthrough = {
                    "type": breakthrough_type,
                    "technology": tech_field,
                    "country": breakthrough_country,
                    "year": current_time.year,
                    "impact": impact,
                    "description": self._generate_breakthrough_description(tech_field, breakthrough_type)
                }
                
                breakthroughs.append(breakthrough)
        
        return breakthroughs
    
    def _calculate_breakthrough_probability(self, tech_field: str,
                                          tech_data: Dict[str, Any],
                                          funding_data: Dict[str, float],
                                          rng: random.Random) -> float:
        """
        计算技术突破概率
        
        Args:
            tech_field: 技术领域
            tech_data: 技术数据
            funding_data: 研究投入数据
            rng: 随机数生成器
            
        Returns:
            突破概率
        """
        # 基础概率
        major_prob = self.innovation_params["breakthrough_probability"]
        inc_prob = self.innovation_params["incremental_probability"]
        
        # 基于成熟度调整概率
        maturity = tech_data["maturity"]
        
        # 中等成熟度（0.3-0.7）的技术更容易有突破
        if 0.3 <= maturity <= 0.7:
            maturity_factor = 1.5
        elif maturity < 0.3:
            maturity_factor = 0.8  # 早期技术还在积累阶段
        else:
            maturity_factor = 0.6  # 成熟技术突破机会减少
        
        # 基于研究投入调整概率
        funding_factor = 1.0
        leading_countries = self._get_leading_countries(tech_field, funding_data)
        if leading_countries:
            avg_funding = sum(funding_data.get(c, 0) for c in leading_countries) / len(leading_countries)
            funding_factor = avg_funding / 3.0  # 相对于3%的基准投入
        
        # 总概率
        total_prob = (major_prob * 0.3 + inc_prob * 0.7) * maturity_factor * funding_factor
        
        return min(total_prob, 0.3)  # 上限30%
    
    def _get_leading_countries(self, tech_field: str, funding_data: Dict[str, float]) -> List[str]:
        """
        获取在特定技术领域领先的国家
        
        Args:
            tech_field: 技术领域
            funding_data: 研究投入数据
            
        Returns:
            领先国家列表
        """
        # 首先获取在该技术领域有优势的国家
        tech_advantage_countries = []
        for country, techs in self.tech_leaders.items():
            if tech_field in techs:
                tech_advantage_countries.append(country)
        
        # 然后按研究投入排序，取前几个
        if tech_advantage_countries:
            sorted_countries = sorted(
                tech_advantage_countries,
                key=lambda c: funding_data.get(c, 0),
                reverse=True
            )
            return sorted_countries[:3]  # 返回前3个
        
        return []
    
    def _get_breakthrough_impact(self, breakthrough_type: str,
                               tech_data: Dict[str, Any]) -> float:
        """
        获取技术突破的影响
        
        Args:
            breakthrough_type: 突破类型
            tech_data: 技术数据
            
        Returns:
            影响值（0-1）
        """
        base_impact = {
            "major": 0.3,
            "significant": 0.15,
            "incremental": 0.05
        }.get(breakthrough_type, 0.1)
        
        # 基于技术潜力调整影响
        potential_factor = tech_data["potential"]
        
        # 基于当前成熟度调整影响
        maturity = tech_data["maturity"]
        if maturity < 0.3:
            maturity_factor = 1.2  # 早期技术突破影响更大
        elif maturity > 0.7:
            maturity_factor = 0.8  # 成熟技术突破影响相对较小
        else:
            maturity_factor = 1.0
        
        total_impact = base_impact * potential_factor * maturity_factor
        return min(total_impact, 0.5)  # 上限50%
    
    def _generate_breakthrough_description(self, tech_field: str,
                                        breakthrough_type: str) -> str:
        """
        生成技术突破描述
        
        Args:
            tech_field: 技术领域
            breakthrough_type: 突破类型
            
        Returns:
            描述文本
        """
        descriptions = {
            "artificial_intelligence": {
                "major": f"{tech_field.replace('_', ' ').title()} 领域实现重大突破，可能带来范式转变",
                "significant": f"{tech_field.replace('_', ' ').title()} 领域取得重要进展，性能大幅提升",
                "incremental": f"{tech_field.replace('_', ' ').title()} 技术获得渐进式改进"
            },
            "quantum_computing": {
                "major": f"量子计算实现量子霸权的新突破，计算能力显著提升",
                "significant": f"量子计算在稳定性或比特数量上取得重要进展",
                "incremental": f"量子计算算法或硬件获得渐进式优化"
            },
            "biotechnology": {
                "major": f"生物技术领域出现革命性突破，可能彻底改变医疗或农业",
                "significant": f"生物技术在特定应用领域取得重要进展",
                "incremental": f"生物技术在现有技术基础上获得改进"
            },
            "renewable_energy": {
                "major": f"可再生能源效率实现跨越式提升，成本大幅下降",
                "significant": f"可再生能源技术取得重要突破，效率明显提高",
                "incremental": f"可再生能源系统获得优化和改进"
            },
            "fusion_energy": {
                "major": f"聚变能源研究获得突破性进展，接近实用化",
                "significant": f"聚变反应维持时间或能量输出取得重要突破",
                "incremental": f"聚变能源技术获得渐进式改进"
            }
        }
        
        # 获取特定技术的描述，如果没有则使用通用描述
        if tech_field in descriptions and breakthrough_type in descriptions[tech_field]:
            return descriptions[tech_field][breakthrough_type]
        else:
            type_prefix = {
                "major": "重大突破：",
                "significant": "重要进展：",
                "incremental": "渐进式改进："
            }.get(breakthrough_type, "")
            return f"{type_prefix}{tech_field.replace('_', ' ').title()} 领域取得技术进展"
    
    def _apply_breakthroughs(self, technology_state: Dict[str, Any],
                           breakthroughs: List[Dict[str, Any]]):
        """
        应用技术突破的影响
        
        Args:
            technology_state: 技术状态
            breakthroughs: 技术突破列表
        """
        technologies = technology_state["technologies"]
        
        for breakthrough in breakthroughs:
            tech_field = breakthrough["technology"]
            impact = breakthrough["impact"]
            
            if tech_field in technologies:
                tech_data = technologies[tech_field]
                
                # 增加技术成熟度
                tech_data["maturity"] = min(1.0, tech_data["maturity"] + impact)
                
                # 提高增长率
                tech_data["growth_rate"] = min(0.2, tech_data["growth_rate"] + impact * 0.3)
                
                # 更新最后突破时间
                tech_data["last_breakthrough"] = breakthrough["year"]
                
                # 可能减少采用挑战
                tech_data["adoption_challenges"] = max(0, tech_data["adoption_challenges"] - impact * 0.2)
    
    def _update_technology_maturity(self, technologies: Dict[str, Dict[str, Any]],
                                   funding_data: Dict[str, float],
                                   rng: random.Random) -> Dict[str, Dict[str, Any]]:
        """
        更新技术成熟度
        
        Args:
            technologies: 技术数据
            funding_data: 研究投入数据
            rng: 随机数生成器
            
        Returns:
            更新后的技术数据
        """
        new_technologies = {}
        
        for tech_field, tech_data in technologies.items():
            new_tech_data = tech_data.copy()
            
            # 基础增长
            base_growth = tech_data["growth_rate"] * rng.uniform(0.8, 1.2)
            
            # 研究投入影响
            leading_countries = self._get_leading_countries(tech_field, funding_data)
            funding_impact = 0
            if leading_countries:
                avg_funding = sum(funding_data.get(c, 0) for c in leading_countries) / len(leading_countries)
                funding_impact = (avg_funding / 3.0 - 1) * 0.01  # 相对于3%的基准
            
            # 随机波动
            random_factor = rng.uniform(-0.01, 0.01)
            
            # 总成熟度增长
            maturity_growth = base_growth + funding_impact + random_factor
            
            # 更新成熟度
            new_tech_data["maturity"] = min(1.0, new_tech_data["maturity"] + maturity_growth)
            
            # 技术成熟后增长率下降
            if new_tech_data["maturity"] > 0.8:
                new_tech_data["growth_rate"] = max(0.01, new_tech_data["growth_rate"] * 0.95)
            
            # 技术淘汰风险（成熟度达到高峰后可能开始下降）
            obsolescence_prob = self.innovation_params["obsolescence_rate"]
            if new_tech_data["maturity"] > 0.9 and rng.random() < obsolescence_prob:
                new_tech_data["maturity"] = max(0.7, new_tech_data["maturity"] - rng.uniform(0.01, 0.03))
            
            new_technologies[tech_field] = new_tech_data
        
        return new_technologies
    
    def _update_country_capabilities(self, country_capabilities: Dict[str, Dict[str, float]],
                                   funding_data: Dict[str, float],
                                   technologies: Dict[str, Dict[str, Any]],
                                   rng: random.Random) -> Dict[str, Dict[str, float]]:
        """
        更新国家技术能力
        
        Args:
            country_capabilities: 当前国家技术能力
            funding_data: 研究投入数据
            technologies: 技术数据
            rng: 随机数生成器
            
        Returns:
            更新后的国家技术能力
        """
        new_capabilities = {}
        
        # 确保所有主要技术国家都有能力数据
        for country in self.tech_leaders.keys():
            if country not in country_capabilities:
                country_capabilities[country] = {}
            
            country_techs = self.tech_leaders[country]
            country_cap = country_capabilities[country].copy()
            
            # 初始化和更新该国家在各技术领域的能力
            for tech_field in technologies.keys():
                # 如果是该国家的优势技术，初始能力较高
                if tech_field in country_techs:
                    if tech_field not in country_cap:
                        country_cap[tech_field] = 0.6 + rng.uniform(-0.1, 0.1)
                else:
                    # 非优势技术初始能力较低
                    if tech_field not in country_cap:
                        country_cap[tech_field] = 0.3 + rng.uniform(-0.1, 0.1)
                
                # 更新技术能力
                current_cap = country_cap[tech_field]
                tech_maturity = technologies[tech_field]["maturity"]
                country_funding = funding_data.get(country, 2.0)  # 默认2%
                
                # 技术能力增长基于技术成熟度和国家研究投入
                growth_rate = (tech_maturity * 0.02) * (country_funding / 3.0)  # 相对于3%的基准
                
                # 随机波动
                growth_rate *= rng.uniform(0.8, 1.2)
                
                # 更新能力
                country_cap[tech_field] = min(1.0, current_cap + growth_rate)
            
            new_capabilities[country] = country_cap
        
        return new_capabilities
    
    def _update_technology_adoption(self, adoption_data: Dict[str, Dict[str, float]],
                                   technologies: Dict[str, Dict[str, Any]],
                                   country_capabilities: Dict[str, Dict[str, float]],
                                   rng: random.Random) -> Dict[str, Dict[str, float]]:
        """
        更新技术采用率
        
        Args:
            adoption_data: 当前技术采用率
            technologies: 技术数据
            country_capabilities: 国家技术能力
            rng: 随机数生成器
            
        Returns:
            更新后的技术采用率
        """
        new_adoption = {}
        diffusion_rate = self.innovation_params["diffusion_rate"]
        
        # 模拟主要国家的技术采用
        for country, capabilities in country_capabilities.items():
            if country not in adoption_data:
                adoption_data[country] = {}
            
            country_adoption = adoption_data[country].copy()
            
            for tech_field, tech_data in technologies.items():
                # 初始化采用率
                if tech_field not in country_adoption:
                    # 基于技术成熟度和国家能力的初始采用率
                    base_adoption = tech_data["maturity"] * capabilities.get(tech_field, 0.5) * 0.5
                    country_adoption[tech_field] = base_adoption
                
                # 更新采用率
                current_adoption = country_adoption[tech_field]
                tech_maturity = tech_data["maturity"]
                country_capability = capabilities.get(tech_field, 0.5)
                adoption_challenges = tech_data["adoption_challenges"]
                
                # 计算采用增长率
                # 采用S型曲线模型（简化）
                if current_adoption < 0.1:
                    # 早期采用阶段，增长较慢
                    growth_factor = 0.3
                elif current_adoption < 0.5:
                    # 快速增长阶段
                    growth_factor = 1.0
                elif current_adoption < 0.9:
                    # 放缓增长阶段
                    growth_factor = 0.5
                else:
                    # 接近饱和
                    growth_factor = 0.1
                
                # 基于多种因素的增长率
                growth_rate = diffusion_rate * growth_factor * country_capability * (1 - adoption_challenges)
                
                # 随机波动
                growth_rate *= rng.uniform(0.8, 1.2)
                
                # 更新采用率
                country_adoption[tech_field] = min(1.0, current_adoption + growth_rate)
            
            new_adoption[country] = country_adoption
        
        return new_adoption
    
    def calculate_technology_readiness(self, technology_state: Dict[str, Any]) -> Dict[str, float]:
        """
        计算各技术领域的就绪度
        
        Args:
            technology_state: 技术状态
            
        Returns:
            各技术领域的就绪度得分（0-1）
        """
        readiness = {}
        technologies = technology_state.get("technologies", {})
        adoption = technology_state.get("technology_adoption", {})
        
        for tech_field, tech_data in technologies.items():
            maturity = tech_data["maturity"]
            
            # 计算平均采用率
            avg_adoption = 0
            adoption_count = 0
            for country, tech_adoption in adoption.items():
                if tech_field in tech_adoption:
                    avg_adoption += tech_adoption[tech_field]
                    adoption_count += 1
            
            if adoption_count > 0:
                avg_adoption /= adoption_count
            else:
                avg_adoption = tech_data["maturity"] * 0.3  # 估计值
            
            # 就绪度 = 成熟度 * 0.6 + 采用率 * 0.4
            readiness[tech_field] = maturity * 0.6 + avg_adoption * 0.4
        
        return readiness


# 测试代码
if __name__ == "__main__":
    # 简单的测试配置
    config = {
        "research": {
            "baseline_funding_growth": 0.03,
            "volatility": 0.02
        },
        "innovation": {
            "breakthrough_probability": 0.02,
            "incremental_probability": 0.15
        }
    }
    
    # 创建技术模型
    model = TechnologyModel(config)
    
    # 初始技术状态
    initial_state = {
        "technologies": {},
        "research_funding": {
            "US": 3.4,
            "China": 2.4,
            "EU": 2.3,
            "Japan": 3.2,
            "South_Korea": 4.8,
            "Israel": 5.4,
            "Germany": 3.1
        },
        "breakthroughs": [],
        "country_capabilities": {},
        "technology_adoption": {}
    }
    
    # 模拟演化
    from datetime import datetime
    
    print("初始技术状态:")
    print(f"研究投入: {initial_state['research_funding']}")
    
    # 演化一次
    new_state = model.evolve(initial_state, datetime.now())
    
    print("\n演化后的技术状态:")
    print(f"技术突破: {new_state['breakthroughs']}")
    print(f"\n示例技术成熟度:")
    for tech, data in list(new_state['technologies'].items())[:5]:  # 只显示前5个
        print(f"{tech}: {data['maturity']:.4f}")
    
    # 计算技术就绪度
    readiness = model.calculate_technology_readiness(new_state)
    print(f"\n示例技术就绪度:")
    for tech, score in list(readiness.items())[:5]:  # 只显示前5个
        print(f"{tech}: {score:.4f}")