#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
气候模型模块
负责模拟全球气候变化和环境因素
"""

import logging
import random
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import math

from ..utils.logger import get_logger
from ..utils.random_utils import get_seeded_random


class ClimateModel:
    """
    气候模型
    模拟全球气候变化和环境因素
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化气候模型
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("ClimateModel")
        self.config = config
        
        # 气候参数配置
        self.emissions_params = config.get("emissions", {
            "baseline_growth_rate": 0.01,   # 基线排放增长率
            "policy_impact": 0.3,          # 政策对减排的影响
            "technology_impact": 0.25,     # 技术对减排的影响
            "economic_impact": 0.2         # 经济对排放的影响
        })
        
        self.temperature_params = config.get("temperature", {
            "climate_sensitivity": 3.0,     # 气候敏感性（℃/2xCO2）
            "transient_response": 0.7,     # 瞬态响应系数
            "volcanic_impact": 0.1,        # 火山喷发影响
            "solar_impact": 0.05           # 太阳活动影响
        })
        
        self.extreme_events_params = config.get("extreme_events", {
            "hurricane_probability": 0.2,  # 飓风概率
            "drought_probability": 0.3,    # 干旱概率
            "flood_probability": 0.25,     # 洪水概率
            "wildfire_probability": 0.15,  # 野火概率
            "temperature_sensitivity": 0.5 # 温度对极端事件的敏感度
        })
        
        # 地区定义
        self.regions = config.get("regions", [
            "North_America",
            "South_America",
            "Europe",
            "Africa",
            "Asia",
            "Oceania",
            "Arctic",
            "Antarctica"
        ])
        
        # 部门定义
        self.sectors = config.get("sectors", [
            "energy",
            "industry",
            "agriculture",
            "transportation",
            "buildings",
            "forestry",
            "waste"
        ])
        
        # 碳汇配置
        self.carbon_sinks = config.get("carbon_sinks", {
            "oceans": {
                "capacity": 30,              # 单位：GtCO2/年
                "efficiency": 0.8           # 吸收效率
            },
            "forests": {
                "capacity": 15,              # 单位：GtCO2/年
                "efficiency": 0.6           # 吸收效率
            },
            "soils": {
                "capacity": 10,              # 单位：GtCO2/年
                "efficiency": 0.5           # 吸收效率
            }
        })
        
        # 初始排放数据（单位：GtCO2/年）
        self.initial_emissions = config.get("initial_emissions", {
            "energy": 15.7,
            "industry": 6.2,
            "agriculture": 5.2,
            "transportation": 7.3,
            "buildings": 3.2,
            "forestry": -1.1,
            "waste": 1.6
        })
        
        self.logger.info("气候模型初始化完成")
    
    def initialize_state(self, initial_year: int) -> Dict[str, Any]:
        """
        初始化气候状态
        
        Args:
            initial_year: 初始年份
            
        Returns:
            初始化后的气候状态字典
        """
        # 创建基础气候状态
        climate_state = {
            "year": initial_year,
            "temperature": {},
            "emissions": {},
            "carbon_concentration": {},
            "sea_level": 8.9,  # 单位：英寸上升（相对于1900年）
            "extreme_events": [],
            "policy_impact": 0.1,
            "carbon_budget": 1000  # 剩余碳预算（单位：GtCO2）
        }
        
        # 初始化全球平均温度（相对于工业化前水平）
        climate_state["temperature"]["global"] = 1.2  # 单位：℃
        
        # 初始化地区温度
        for region in self.regions:
            # 地区温度异常，根据地区不同设置不同值
            if region == "Arctic":
                climate_state["temperature"][region] = 3.5
            elif region == "Antarctica":
                climate_state["temperature"][region] = 0.5
            elif region in ["North_America", "Europe", "Asia"]:
                climate_state["temperature"][region] = 1.3
            else:
                climate_state["temperature"][region] = 1.1
        
        # 初始化各部门排放量
        for sector, emission in self.initial_emissions.items():
            climate_state["emissions"][sector] = emission
        
        # 初始化碳浓度
        climate_state["carbon_concentration"]["co2"] = 415.0  # 单位：ppm
        climate_state["carbon_concentration"]["methane"] = 1875.0  # 单位：ppb
        climate_state["carbon_concentration"]["nitrous_oxide"] = 332.0  # 单位：ppb
        
        return climate_state
    
    def evolve(self, climate_state: Dict[str, Any], current_time: datetime, 
              economic_state: Optional[Dict[str, Any]] = None,
              technology_state: Optional[Dict[str, Any]] = None,
              political_state: Optional[Dict[str, Any]] = None,
              rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """
        演化气候状态
        
        Args:
            climate_state: 当前气候状态
            current_time: 当前时间
            economic_state: 经济状态（可选）
            technology_state: 技术状态（可选）
            political_state: 政治状态（可选）
            rng: 随机数生成器
            
        Returns:
            演化后的气候状态
        """
        try:
            # 创建新状态以避免修改原始数据
            new_state = climate_state.copy()
            
            # 确保所有必要的子状态存在
            new_state["emissions"] = new_state.get("emissions", {})
            new_state["temperature"] = new_state.get("temperature", {})
            new_state["carbon_concentration"] = new_state.get("carbon_concentration", 420)  # ppm
            new_state["sea_level"] = new_state.get("sea_level", 0.25)  # 米（相对于1900年）
            new_state["extreme_events"] = new_state.get("extreme_events", [])
            new_state["policies"] = new_state.get("policies", [])
            new_state["carbon_budget"] = new_state.get("carbon_budget", {
                "total": 5000,  # 总碳预算（GtCO2）
                "used": 2400    # 已使用（GtCO2）
            })
            new_state["ecosystems"] = new_state.get("ecosystems", {})
            
            # 初始化随机数生成器
            if rng is None:
                rng = get_seeded_random(hash(str(current_time)))
            
            # 初始化排放数据（如果不存在）
            if not new_state["emissions"]:
                new_state["emissions"] = self.initial_emissions.copy()
            
            # 生成气候政策
            new_policies = self._generate_climate_policies(
                new_state, political_state, current_time, rng
            )
            new_state["policies"].extend(new_policies)
            
            # 生成极端气候事件
            new_events = self._generate_extreme_events(
                new_state, current_time, rng
            )
            new_state["extreme_events"].extend(new_events)
            
            # 更新排放量
            new_state["emissions"] = self._update_emissions(
                new_state["emissions"], new_policies, economic_state,
                technology_state, rng
            )
            
            # 更新碳浓度
            new_state["carbon_concentration"] = self._update_carbon_concentration(
                new_state["carbon_concentration"], new_state["emissions"]
            )
            
            # 更新温度
            new_state["temperature"] = self._update_temperature(
                new_state["temperature"], new_state["carbon_concentration"],
                new_events, rng
            )
            
            # 更新海平面
            new_state["sea_level"] = self._update_sea_level(
                new_state["sea_level"], new_state["temperature"]
            )
            
            # 更新碳预算
            new_state["carbon_budget"] = self._update_carbon_budget(
                new_state["carbon_budget"], new_state["emissions"]
            )
            
            # 更新生态系统状态
            new_state["ecosystems"] = self._update_ecosystems(
                new_state["ecosystems"], new_state["temperature"],
                new_state["emissions"], new_events, rng
            )
            
            return new_state
            
        except Exception as e:
            self.logger.error(f"演化气候状态失败: {str(e)}")
            return climate_state  # 失败时返回原始状态
    
    def _generate_climate_policies(self, climate_state: Dict[str, Any],
                                 political_state: Optional[Dict[str, Any]],
                                 current_time: datetime,
                                 rng: random.Random) -> List[Dict[str, Any]]:
        """
        生成气候政策
        
        Args:
            climate_state: 气候状态
            political_state: 政治状态（可选）
            current_time: 当前时间
            rng: 随机数生成器
            
        Returns:
            生成的政策列表
        """
        policies = []
        
        # 基于政治稳定性和温度变化生成政策
        temp_increase = climate_state.get("temperature", {}).get("global_mean", 1.2)
        
        # 温度越高，越可能出台激进政策
        policy_probability = 0.1 + (temp_increase - 1.0) * 0.1
        
        # 政治稳定性影响政策实施概率
        if political_state and "stability" in political_state:
            stability = political_state["stability"]
            avg_stability = sum(stability.values()) / len(stability) if stability else 0.7
            policy_probability *= avg_stability  # 政治越稳定，越可能实施政策
        
        # 限制最大概率
        policy_probability = min(policy_probability, 0.3)
        
        # 决定是否生成新政策
        if rng.random() < policy_probability:
            policy_type = rng.choice(self.policy_types)
            
            # 确定政策强度
            if rng.random() < 0.3:
                strength = "strong"
                emission_reduction = rng.uniform(0.1, 0.25)
            else:
                strength = "moderate"
                emission_reduction = rng.uniform(0.03, 0.1)
            
            # 确定适用部门
            if policy_type in ["carbon_tax", "cap_and_trade"]:
                sectors = random.sample(self.sectors, k=random.randint(2, len(self.sectors)))
            elif policy_type == "renewable_subsidies":
                sectors = ["energy"]
            elif policy_type == "efficiency_standards":
                sectors = random.sample(["buildings", "transportation", "industry"], k=random.randint(1, 3))
            elif policy_type == "reforestation":
                sectors = ["forestry"]
            elif policy_type == "carbon_capture":
                sectors = ["energy", "industry"]
            else:
                sectors = random.sample(self.sectors, k=random.randint(1, 3))
            
            policy = {
                "type": policy_type,
                "strength": strength,
                "sectors": sectors,
                "emission_reduction": emission_reduction,
                "year": current_time.year,
                "description": self._generate_policy_description(policy_type, strength, sectors)
            }
            
            policies.append(policy)
        
        return policies
    
    def _generate_policy_description(self, policy_type: str, strength: str,
                                   sectors: List[str]) -> str:
        """
        生成政策描述
        
        Args:
            policy_type: 政策类型
            strength: 政策强度
            sectors: 适用部门
            
        Returns:
            政策描述文本
        """
        type_names = {
            "carbon_tax": "碳税",
            "cap_and_trade": "碳排放交易体系",
            "renewable_subsidies": "可再生能源补贴",
            "efficiency_standards": "能效标准",
            "reforestation": "植树造林计划",
            "carbon_capture": "碳捕获技术支持"
        }
        
        strength_adjectives = {
            "strong": "强有力的",
            "moderate": "中等强度的",
            "weak": "温和的"
        }
        
        sector_names = {
            "energy": "能源",
            "industry": "工业",
            "agriculture": "农业",
            "transportation": "交通",
            "buildings": "建筑",
            "forestry": "林业",
            "waste": "废弃物"
        }
        
        sector_text = "、".join([sector_names.get(s, s) for s in sectors])
        
        return f"实施{strength_adjectives.get(strength, '')}{type_names.get(policy_type, policy_type)}政策，覆盖{sector_text}部门"
    
    def _generate_extreme_events(self, climate_state: Dict[str, Any],
                               current_time: datetime,
                               rng: random.Random) -> List[Dict[str, Any]]:
        """
        生成极端气候事件
        
        Args:
            climate_state: 气候状态
            current_time: 当前时间
            rng: 随机数生成器
            
        Returns:
            生成的极端事件列表
        """
        events = []
        temp_increase = climate_state.get("temperature", {}).get("global_mean", 1.2)
        
        # 温度升高增加极端事件概率
        temp_factor = 1.0 + (temp_increase - 1.0) * self.extreme_events_params["temperature_sensitivity"]
        
        # 检查各种极端事件
        event_types = [
            ("hurricane", self.extreme_events_params["hurricane_probability"]),
            ("drought", self.extreme_events_params["drought_probability"]),
            ("flood", self.extreme_events_params["flood_probability"]),
            ("wildfire", self.extreme_events_params["wildfire_probability"])
        ]
        
        for event_type, base_prob in event_types:
            # 调整概率
            adjusted_prob = base_prob * temp_factor
            
            if rng.random() < adjusted_prob:
                # 选择地区
                region = rng.choice(self.regions)
                
                # 确定严重程度
                severity = rng.choice(["minor", "moderate", "major", "extreme"])
                severity_factor = {
                    "minor": 1.0,
                    "moderate": 2.0,
                    "major": 4.0,
                    "extreme": 8.0
                }[severity]
                
                # 计算影响
                economic_impact = rng.uniform(1, 10) * severity_factor  # 十亿美元
                death_toll = rng.randint(1, 1000) * math.floor(severity_factor)
                
                event = {
                    "type": event_type,
                    "region": region,
                    "severity": severity,
                    "year": current_time.year,
                    "economic_impact": economic_impact,
                    "death_toll": death_toll,
                    "description": self._generate_event_description(event_type, region, severity)
                }
                
                events.append(event)
        
        # 可能的火山喷发（降温事件）
        if rng.random() < 0.02:
            event = {
                "type": "volcanic_eruption",
                "region": rng.choice(["Indonesia", "Iceland", "Philippines", "Chile"]),
                "severity": rng.choice(["moderate", "major"]),
                "year": current_time.year,
                "cooling_impact": rng.uniform(0.1, 0.5),  # 降温度数
                "description": f"{rng.choice(["Indonesia", "Iceland", "Philippines", "Chile"])}发生火山喷发"
            }
            events.append(event)
        
        return events
    
    def _generate_event_description(self, event_type: str, region: str,
                                  severity: str) -> str:
        """
        生成极端事件描述
        
        Args:
            event_type: 事件类型
            region: 地区
            severity: 严重程度
            
        Returns:
            事件描述文本
        """
        type_names = {
            "hurricane": "飓风",
            "drought": "干旱",
            "flood": "洪水",
            "wildfire": "野火"
        }
        
        severity_adjectives = {
            "minor": "轻微的",
            "moderate": "中等的",
            "major": "严重的",
            "extreme": "极端的"
        }
        
        return f"{region}发生{severity_adjectives.get(severity, '')}{type_names.get(event_type, event_type)}"
    
    def _update_emissions(self, emissions: Dict[str, float],
                         policies: List[Dict[str, Any]],
                         economic_state: Optional[Dict[str, Any]],
                         technology_state: Optional[Dict[str, Any]],
                         rng: random.Random) -> Dict[str, float]:
        """
        更新排放量
        
        Args:
            emissions: 当前排放量
            policies: 气候政策
            economic_state: 经济状态（可选）
            technology_state: 技术状态（可选）
            rng: 随机数生成器
            
        Returns:
            更新后的排放量
        """
        new_emissions = emissions.copy()
        baseline_growth = self.emissions_params["baseline_growth_rate"]
        
        # 计算各部门的排放变化
        for sector, current_emission in new_emissions.items():
            # 基础变化（考虑经济增长）
            growth_factor = 1.0 + baseline_growth
            
            # 经济影响
            if economic_state:
                economic_factor = self._calculate_economic_emission_impact(
                    sector, economic_state, rng
                )
                growth_factor += economic_factor
            
            # 技术影响
            tech_reduction = 0.0
            if technology_state:
                tech_reduction = self._calculate_technology_emission_reduction(
                    sector, technology_state
                )
            
            # 政策影响
            policy_reduction = 0.0
            for policy in policies:
                if sector in policy["sectors"]:
                    policy_reduction += policy["emission_reduction"] * self.emissions_params["policy_impact"]
            
            # 随机波动
            random_factor = rng.uniform(-0.02, 0.02)
            
            # 计算净变化
            net_factor = (growth_factor + random_factor) * (1 - tech_reduction) * (1 - policy_reduction)
            
            # 更新排放
            new_emissions[sector] = current_emission * net_factor
            
            # 确保林业等负排放部门保持合理范围
            if sector == "forestry" and new_emissions[sector] > 0:
                new_emissions[sector] = -abs(new_emissions[sector]) * 0.3
        
        return new_emissions
    
    def _calculate_economic_emission_impact(self, sector: str,
                                          economic_state: Dict[str, Any],
                                          rng: random.Random) -> float:
        """
        计算经济对排放的影响
        
        Args:
            sector: 部门
            economic_state: 经济状态
            rng: 随机数生成器
            
        Returns:
            经济影响值
        """
        impact = 0
        sensitivity = self.emissions_params["economic_impact"]
        
        # 简化模型：GDP增长通常增加排放
        # 但经济结构变化可能降低排放强度
        gdp_data = economic_state.get("gdp", {})
        
        if gdp_data:
            # 假设全球GDP增长
            total_gdp = sum(gdp_data.values())
            # 模拟GDP增长率（2-4%）
            gdp_growth = rng.uniform(0.02, 0.04)
            
            # 不同部门对经济增长的敏感度不同
            sector_sensitivity = {
                "energy": 0.8,
                "industry": 0.7,
                "agriculture": 0.5,
                "transportation": 0.6,
                "buildings": 0.4,
                "forestry": -0.1,  # 经济增长可能增加林业投入
                "waste": 0.5
            }.get(sector, 0.5)
            
            # 排放强度变化（随时间降低）
            intensity_reduction = rng.uniform(0.01, 0.03)
            
            impact = (gdp_growth * sector_sensitivity - intensity_reduction) * sensitivity
        
        return impact
    
    def _calculate_technology_emission_reduction(self, sector: str,
                                              technology_state: Dict[str, Any]) -> float:
        """
        计算技术对减排的影响
        
        Args:
            sector: 部门
            technology_state: 技术状态
            
        Returns:
            减排比例
        """
        reduction = 0
        sensitivity = self.emissions_params["technology_impact"]
        
        # 检查相关技术的成熟度
        tech_maturity = technology_state.get("technologies", {})
        
        # 部门与技术的映射
        sector_technologies = {
            "energy": ["renewable_energy", "fusion_energy"],
            "industry": ["advanced_materials", "digital_twins"],
            "transportation": ["autonomous_systems"],
            "buildings": ["digital_twins"],
            "forestry": [],
            "waste": [],
            "agriculture": ["biotechnology"]
        }
        
        relevant_technologies = sector_technologies.get(sector, [])
        
        if relevant_technologies:
            # 计算相关技术的平均成熟度
            total_maturity = 0
            count = 0
            
            for tech in relevant_technologies:
                if tech in tech_maturity:
                    total_maturity += tech_maturity[tech]["maturity"]
                    count += 1
            
            if count > 0:
                avg_maturity = total_maturity / count
                # 技术成熟度越高，减排效果越好
                reduction = avg_maturity * sensitivity
        
        return reduction
    
    def _update_carbon_concentration(self, current_concentration: float,
                                   emissions: Dict[str, float]) -> float:
        """
        更新大气碳浓度
        
        Args:
            current_concentration: 当前碳浓度（ppm）
            emissions: 排放量（GtCO2/年）
            
        Returns:
            更新后的碳浓度
        """
        # 计算总排放量
        total_emissions = sum(emissions.values())
        
        # 简化模型：每年排放的CO2一部分留在大气中
        # 大约45%的排放会留在大气中（简化）
        atmospheric_retention = 0.45
        emissions_to_atmosphere = total_emissions * atmospheric_retention
        
        # 转换GtCO2到ppm变化
        # 1 ppm CO2 ≈ 7.8 GtCO2
        concentration_increase = emissions_to_atmosphere / 7.8
        
        # 更新浓度
        new_concentration = current_concentration + concentration_increase
        
        # 考虑碳汇的吸收（简化模型）
        # 假设碳汇每年可以吸收一定比例的增加量
        sink_efficiency = 0.3  # 30%的新增排放被碳汇吸收
        new_concentration = current_concentration + concentration_increase * (1 - sink_efficiency)
        
        return new_concentration
    
    def _update_temperature(self, temperature: Dict[str, float],
                           carbon_concentration: float,
                           extreme_events: List[Dict[str, Any]],
                           rng: random.Random) -> Dict[str, float]:
        """
        更新温度
        
        Args:
            temperature: 当前温度数据
            carbon_concentration: 碳浓度（ppm）
            extreme_events: 极端事件
            rng: 随机数生成器
            
        Returns:
            更新后的温度数据
        """
        new_temperature = temperature.copy()
        
        # 确保全球平均温度存在
        if "global_mean" not in new_temperature:
            new_temperature["global_mean"] = 1.2  # 相对于工业化前水平
        
        # 计算温室气体导致的温度上升
        # 使用简化的气候模型
        # 参考水平（工业化前）约280 ppm
        reference_concentration = 280
        
        # 计算辐射强迫
        # ΔF = 5.35 * ln(C/C0)  W/m²
        radiative_forcing = 5.35 * math.log(carbon_concentration / reference_concentration)
        
        # 计算平衡温度变化
        # ΔTeq = λ * ΔF
        climate_sensitivity = self.temperature_params["climate_sensitivity"]
        equilibrium_warming = climate_sensitivity * (radiative_forcing / 3.7)  # 标准化到2xCO2
        
        # 计算瞬态温度响应（气候系统惯性）
        transient_response = self.temperature_params["transient_response"]
        current_warming = new_temperature["global_mean"]
        
        # 温度向平衡态接近
        temperature_increase = (equilibrium_warming - current_warming) * transient_response * 0.1
        
        # 随机气候波动
        random_fluctuation = rng.uniform(-0.05, 0.05)
        
        # 极端事件影响
        event_impact = 0
        for event in extreme_events:
            if event["type"] == "volcanic_eruption":
                # 火山喷发导致降温
                event_impact -= event.get("cooling_impact", 0) * self.temperature_params["volcanic_impact"]
        
        # 太阳活动影响
        solar_variation = rng.uniform(-0.03, 0.03) * self.temperature_params["solar_impact"]
        
        # 总温度变化
        total_change = temperature_increase + random_fluctuation + event_impact + solar_variation
        
        # 更新全球平均温度
        new_temperature["global_mean"] = max(0.5, new_temperature["global_mean"] + total_change)
        
        # 更新地区温度（简化，假设地区温度变化与全球平均相关）
        for region in self.regions:
            if region not in new_temperature:
                # 高纬度地区变暖更快，极地放大效应
                if region in ["Arctic", "Antarctica"]:
                    base_warming = new_temperature["global_mean"] * 2.0  # 极地放大效应
                else:
                    base_warming = new_temperature["global_mean"] * (0.8 + rng.uniform(0, 0.4))
                new_temperature[region] = base_warming
            else:
                # 地区温度变化与全球平均相关，但有区域差异
                regional_factor = 1.0
                if region in ["Arctic", "Antarctica"]:
                    regional_factor = 1.5  # 极地放大效应
                new_temperature[region] += total_change * regional_factor
        
        return new_temperature
    
    def _update_sea_level(self, current_sea_level: float,
                         temperature: Dict[str, float]) -> float:
        """
        更新海平面
        
        Args:
            current_sea_level: 当前海平面上升（米，相对于1900年）
            temperature: 温度数据
            
        Returns:
            更新后的海平面
        """
        # 简化模型：海平面上升与全球温度相关
        global_temp = temperature.get("global_mean", 1.2)
        
        # 当前约上升0.25米（相对于1900年）
        # 假设温度每升高1°C，海平面最终将上升约0.5-2米
        # 但有滞后效应
        
        # 计算海平面上升速率
        # 温度越高，上升越快
        if global_temp < 1.5:
            rate = 0.005  # 每年5毫米
        elif global_temp < 2.0:
            rate = 0.008  # 每年8毫米
        elif global_temp < 3.0:
            rate = 0.012  # 每年12毫米
        else:
            rate = 0.015  # 每年15毫米
        
        new_sea_level = current_sea_level + rate
        
        return new_sea_level
    
    def _update_carbon_budget(self, carbon_budget: Dict[str, float],
                             emissions: Dict[str, float]) -> Dict[str, float]:
        """
        更新碳预算
        
        Args:
            carbon_budget: 碳预算数据
            emissions: 排放量
            
        Returns:
            更新后的碳预算
        """
        new_budget = carbon_budget.copy()
        
        # 计算年排放量
        total_emissions = sum(emissions.values())
        
        # 更新已使用预算
        new_budget["used"] += total_emissions
        
        return new_budget
    
    def _update_ecosystems(self, ecosystems: Dict[str, Any],
                          temperature: Dict[str, float],
                          emissions: Dict[str, float],
                          extreme_events: List[Dict[str, Any]],
                          rng: random.Random) -> Dict[str, Any]:
        """
        更新生态系统状态
        
        Args:
            ecosystems: 当前生态系统状态
            temperature: 温度数据
            emissions: 排放量
            extreme_events: 极端事件
            rng: 随机数生成器
            
        Returns:
            更新后的生态系统状态
        """
        new_ecosystems = ecosystems.copy()
        
        # 主要生态系统类型
        ecosystem_types = [
            "coral_reefs",
            "tropical_forests",
            "arctic_tundra",
            "glaciers",
            "marine_ecosystems",
            "grasslands",
            "biodiversity"
        ]
        
        # 初始化生态系统状态
        for eco_type in ecosystem_types:
            if eco_type not in new_ecosystems:
                # 初始健康状态（0-1）
                initial_health = self._get_initial_ecosystem_health(eco_type)
                new_ecosystems[eco_type] = {
                    "health": initial_health,
                    "last_change": 0,
                    "resilience": 0.3 + rng.uniform(0, 0.4)
                }
        
        global_temp = temperature.get("global_mean", 1.2)
        
        # 更新每个生态系统
        for eco_type, eco_data in new_ecosystems.items():
            health = eco_data["health"]
            resilience = eco_data["resilience"]
            
            # 温度影响
            temp_impact = self._calculate_temperature_ecosystem_impact(eco_type, global_temp)
            
            # 极端事件影响
            event_impact = 0
            for event in extreme_events:
                event_impact += self._calculate_event_ecosystem_impact(eco_type, event)
            
            # 排放影响（海洋酸化等）
            emission_impact = 0
            if eco_type in ["coral_reefs", "marine_ecosystems"]:
                # 碳排放导致海洋酸化
                total_emissions = sum(emissions.values())
                emission_impact = -total_emissions * 0.0001
            
            # 生态系统恢复能力
            recovery = resilience * 0.02  # 每年恢复一定比例
            
            # 总变化
            total_change = temp_impact + event_impact + emission_impact + recovery
            
            # 更新健康状态
            new_health = max(0, min(1, health + total_change))
            new_ecosystems[eco_type]["health"] = new_health
            new_ecosystems[eco_type]["last_change"] = total_change
        
        return new_ecosystems
    
    def _get_initial_ecosystem_health(self, eco_type: str) -> float:
        """
        获取生态系统的初始健康状态
        
        Args:
            eco_type: 生态系统类型
            
        Returns:
            初始健康状态（0-1）
        """
        initial_health = {
            "coral_reefs": 0.3,      # 珊瑚礁状况较差
            "tropical_forests": 0.6,  # 热带雨林正在减少
            "arctic_tundra": 0.4,    # 北极苔原受到严重威胁
            "glaciers": 0.3,         # 冰川正在退缩
            "marine_ecosystems": 0.5, # 海洋生态系统面临压力
            "grasslands": 0.6,       # 草原状况中等
            "biodiversity": 0.5      # 生物多样性正在下降
        }
        
        return initial_health.get(eco_type, 0.5)
    
    def _calculate_temperature_ecosystem_impact(self, eco_type: str,
                                              global_temp: float) -> float:
        """
        计算温度对生态系统的影响
        
        Args:
            eco_type: 生态系统类型
            global_temp: 全球平均温度
            
        Returns:
            影响值
        """
        # 不同生态系统对温度的敏感度不同
        sensitivity = {
            "coral_reefs": 0.15,      # 对温度非常敏感
            "arctic_tundra": 0.12,    # 对温度高度敏感
            "glaciers": 0.1,          # 对温度敏感
            "tropical_forests": 0.08,  # 中等敏感度
            "marine_ecosystems": 0.06, # 中等敏感度
            "grasslands": 0.04,       # 较低敏感度
            "biodiversity": 0.07      # 中等敏感度
        }.get(eco_type, 0.05)
        
        # 温度越高，负面影响越大
        # 假设1.5°C是安全阈值
        if global_temp > 1.5:
            excess_temp = global_temp - 1.5
            return -excess_temp * sensitivity
        else:
            return 0
    
    def _calculate_event_ecosystem_impact(self, eco_type: str,
                                        event: Dict[str, Any]) -> float:
        """
        计算极端事件对生态系统的影响
        
        Args:
            eco_type: 生态系统类型
            event: 极端事件
            
        Returns:
            影响值
        """
        event_type = event["type"]
        severity = event.get("severity", "moderate")
        
        # 不同事件对不同生态系统的影响
        impact_matrix = {
            "hurricane": {
                "coral_reefs": -0.1,
                "tropical_forests": -0.15,
                "marine_ecosystems": -0.08
            },
            "drought": {
                "tropical_forests": -0.12,
                "grasslands": -0.1,
                "biodiversity": -0.08
            },
            "flood": {
                "grasslands": -0.05,
                "tropical_forests": -0.08
            },
            "wildfire": {
                "tropical_forests": -0.2,
                "grasslands": -0.15,
                "biodiversity": -0.12
            }
        }
        
        # 严重程度因子
        severity_factor = {
            "minor": 0.3,
            "moderate": 0.6,
            "major": 0.9,
            "extreme": 1.2
        }.get(severity, 0.6)
        
        # 获取基础影响
        base_impact = 0
        if event_type in impact_matrix and eco_type in impact_matrix[event_type]:
            base_impact = impact_matrix[event_type][eco_type]
        
        return base_impact * severity_factor
    
    def calculate_climate_risk(self, climate_state: Dict[str, Any]) -> float:
        """
        计算气候风险指数
        
        Args:
            climate_state: 气候状态
            
        Returns:
            0-1之间的风险指数（越高风险越大）
        """
        if not climate_state:
            return 0.5
        
        # 风险因子权重
        weights = {
            "temperature": 0.3,
            "sea_level": 0.2,
            "emissions": 0.2,
            "carbon_budget": 0.15,
            "ecosystems": 0.1,
            "extreme_events": 0.05
        }
        
        risk_score = 0
        
        # 温度风险
        temp = climate_state.get("temperature", {})
        if "global_mean" in temp:
            global_temp = temp["global_mean"]
            # 1.5°C为基准，超过越多风险越大
            if global_temp > 1.5:
                temp_risk = min((global_temp - 1.5) / 2.0, 1.0)  # 4°C时达到最大风险
            else:
                temp_risk = 0
            risk_score += temp_risk * weights["temperature"]
        
        # 海平面风险
        sea_level = climate_state.get("sea_level", 0.25)
        # 0.5米为基准
        sea_risk = min((sea_level - 0.5) / 1.0, 1.0)  # 1.5米时达到最大风险
        sea_risk = max(0, sea_risk)
        risk_score += sea_risk * weights["sea_level"]
        
        # 排放风险
        emissions = climate_state.get("emissions", {})
        total_emissions = sum(emissions.values())
        # 35 GtCO2/年为基准
        emission_risk = min((total_emissions - 35) / 20.0, 1.0)  # 55 GtCO2/年时达到最大风险
        emission_risk = max(0, emission_risk)
        risk_score += emission_risk * weights["emissions"]
        
        # 碳预算风险
        carbon_budget = climate_state.get("carbon_budget", {})
        if "total" in carbon_budget and "used" in carbon_budget:
            budget_percentage = carbon_budget["used"] / carbon_budget["total"]
            budget_risk = min((budget_percentage - 0.5) / 0.5, 1.0)  # 用完预算时达到最大风险
            budget_risk = max(0, budget_risk)
            risk_score += budget_risk * weights["carbon_budget"]
        
        # 生态系统风险
        ecosystems = climate_state.get("ecosystems", {})
        if ecosystems:
            avg_health = sum(eco["health"] for eco in ecosystems.values()) / len(ecosystems)
            ecosystem_risk = 1 - avg_health  # 健康度越低，风险越高
            risk_score += ecosystem_risk * weights["ecosystems"]
        
        # 极端事件风险
        recent_events = [e for e in climate_state.get("extreme_events", []) 
                        if e.get("year") >= datetime.now().year - 5]
        event_risk = min(len(recent_events) / 10.0, 1.0)  # 10个事件时达到最大风险
        risk_score += event_risk * weights["extreme_events"]
        
        return min(1.0, risk_score)  # 确保在0-1范围内


# 测试代码
if __name__ == "__main__":
    # 简单的测试配置
    config = {
        "emissions": {
            "baseline_growth_rate": 0.01,
            "policy_impact": 0.3
        },
        "temperature": {
            "climate_sensitivity": 3.0,
            "transient_response": 0.7
        }
    }
    
    # 创建气候模型
    model = ClimateModel(config)
    
    # 初始气候状态
    initial_state = {
        "emissions": {
            "energy": 15.7,
            "industry": 6.2,
            "agriculture": 5.2,
            "transportation": 8.0,
            "buildings": 3.0,
            "forestry": -1.2,
            "waste": 1.5
        },
        "temperature": {
            "global_mean": 1.2,
            "North_America": 1.3,
            "Europe": 1.4,
            "Asia": 1.1,
            "Arctic": 2.5
        },
        "carbon_concentration": 420,
        "sea_level": 0.25,
        "extreme_events": [],
        "policies": [],
        "carbon_budget": {
            "total": 5000,
            "used": 2400
        },
        "ecosystems": {}
    }
    
    # 模拟演化
    from datetime import datetime
    
    print("初始气候状态:")
    print(f"全球平均温度上升: {initial_state['temperature']['global_mean']}°C")
    print(f"碳浓度: {initial_state['carbon_concentration']} ppm")
    print(f"总排放量: {sum(initial_state['emissions'].values()):.2f} GtCO2/年")
    
    # 演化一次
    new_state = model.evolve(initial_state, datetime.now())
    
    print("\n演化后的气候状态:")
    print(f"全球平均温度上升: {new_state['temperature']['global_mean']:.2f}°C")
    print(f"碳浓度: {new_state['carbon_concentration']:.2f} ppm")
    print(f"总排放量: {sum(new_state['emissions'].values()):.2f} GtCO2/年")
    print(f"海平面上升: {new_state['sea_level']:.3f} 米")
    print(f"极端事件: {new_state['extreme_events']}")
    print(f"新政策: {new_state['policies']}")
    
    # 计算气候风险
    risk_score = model.calculate_climate_risk(new_state)
    print(f"\n气候风险指数: {risk_score:.4f}")