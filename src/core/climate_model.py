import numpy as np
from typing import Dict, List, Any, Optional
import datetime
import logging

class ClimateModel:
    """
    气候领域模型 - 负责模拟全球气候变化、极端天气事件和气候政策影响
    
    核心功能:
    - 气候变化模拟
    - 极端天气事件生成
    - 碳排放和温室气体浓度跟踪
    - 气候政策影响评估
    - 地区脆弱性分析
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化气候模型
        
        Args:
            config: 配置参数
        """
        self.logger = logging.getLogger(__name__)
        
        # 默认配置
        self.default_config = {
            "volatility": 0.1,  # 气候系统波动性
            "base_warming_rate": 0.02,  # 基础全球变暖速率（每年°C）
            "extreme_event_probability": 0.05,  # 极端事件发生概率
            "carbon_emission_impact": 0.005,  # 碳排放对温度的影响系数
            "policy_effectiveness": 0.5,  # 气候政策有效性
            "max_global_temp": 5.0,  # 模型允许的最大全球温度上升
            "min_global_temp": -1.0,  # 模型允许的最小全球温度变化
            "climate_sensitivity": 2.0,  # 气候敏感性参数
            "regions": [
                "North America",
                "South America",
                "Europe",
                "Africa",
                "Asia",
                "Oceania",
                "Antarctica"
            ],
            "region_vulnerability": {},  # 地区脆弱性指数
            "extreme_event_types": [
                "heatwave",
                "flood",
                "drought",
                "hurricane",
                "wildfire",
                "cold_wave",
                "storm"
            ],
            "greenhouse_gases": [
                "CO2",
                "CH4",
                "N2O",
                "F-gases"
            ],
            "co2_equivalence": {  # 温室气体等效系数
                "CO2": 1,
                "CH4": 28,
                "N2O": 265,
                "F-gases": 1000
            }
        }
        
        # 合并配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # 提取配置参数
        self.volatility = self.config["volatility"]
        self.base_warming_rate = self.config["base_warming_rate"]
        self.extreme_event_probability = self.config["extreme_event_probability"]
        self.carbon_emission_impact = self.config["carbon_emission_impact"]
        self.policy_effectiveness = self.config["policy_effectiveness"]
        self.max_global_temp = self.config["max_global_temp"]
        self.min_global_temp = self.config["min_global_temp"]
        self.climate_sensitivity = self.config["climate_sensitivity"]
        self.regions = self.config["regions"]
        self.region_vulnerability = self.config["region_vulnerability"]
        self.extreme_event_types = self.config["extreme_event_types"]
        self.greenhouse_gases = self.config["greenhouse_gases"]
        self.co2_equivalence = self.config["co2_equivalence"]
        
        # 初始化地区脆弱性
        if not self.region_vulnerability:
            self._initialize_region_vulnerability()
        
        self.logger.info("气候模型初始化完成")
    
    def _initialize_region_vulnerability(self):
        """
        初始化地区气候脆弱性
        """
        # 基于现有研究的简化脆弱性指数
        self.region_vulnerability = {
            "North America": 0.4,
            "South America": 0.6,
            "Europe": 0.3,
            "Africa": 0.8,
            "Asia": 0.7,
            "Oceania": 0.5,
            "Antarctica": 0.9
        }
        self.logger.info("地区气候脆弱性初始化完成")
    
    def evolve(self, climate_state: Dict[str, Any],
              human_activities: Dict[str, Any],
              random_state: np.random.RandomState) -> Dict[str, Any]:
        """
        演化气候状态
        
        Args:
            climate_state: 当前气候状态
            human_activities: 人类活动（如碳排放、政策等）
            random_state: 随机数生成器
            
        Returns:
            更新后的气候状态
        """
        # 创建新的气候状态对象
        updated_climate = {}
        for key, value in climate_state.items():
            if isinstance(value, dict):
                updated_climate[key] = value.copy()
            else:
                updated_climate[key] = value
        
        # 演化全球温度
        updated_climate["global_temperature"] = self._evolve_global_temperature(
            climate_state["global_temperature"],
            human_activities.get("carbon_emissions", {}),
            human_activities.get("climate_policies", []),
            random_state
        )
        
        # 演化地区温度
        updated_climate["regional_temperatures"] = self._evolve_regional_temperatures(
            climate_state["regional_temperatures"],
            updated_climate["global_temperature"],
            random_state
        )
        
        # 更新温室气体浓度
        updated_climate["greenhouse_gas_concentrations"] = self._update_greenhouse_gases(
            climate_state["greenhouse_gas_concentrations"],
            human_activities.get("greenhouse_gas_emissions", {})
        )
        
        # 生成极端天气事件
        extreme_events = self._generate_extreme_events(
            updated_climate["regional_temperatures"],
            human_activities.get("climate_policies", []),
            random_state
        )
        updated_climate["recent_extreme_events"] = extreme_events
        
        # 计算海平面上升
        updated_climate["sea_level_rise"] = self._calculate_sea_level_rise(
            updated_climate["global_temperature"]
        )
        
        # 计算气候影响
        updated_climate["climate_impacts"] = self._calculate_climate_impacts(
            updated_climate["regional_temperatures"],
            extreme_events
        )
        
        # 更新时间戳
        updated_climate["timestamp"] = datetime.datetime.now().isoformat()
        
        self.logger.debug("气候状态演化完成")
        return updated_climate
    
    def _evolve_global_temperature(self, current_temp: float,
                                  carbon_emissions: Dict[str, float],
                                  climate_policies: List[Dict[str, Any]],
                                  random_state: np.random.RandomState) -> float:
        """
        演化全球温度
        
        Args:
            current_temp: 当前全球温度变化
            carbon_emissions: 碳排放数据
            climate_policies: 气候政策列表
            random_state: 随机数生成器
            
        Returns:
            新的全球温度变化
        """
        # 基础温度变化
        base_change = self.base_warming_rate
        
        # 碳排放影响
        total_emissions = sum(carbon_emissions.values())
        emission_impact = self.carbon_emission_impact * total_emissions * 0.001  # 缩放因子
        
        # 政策影响
        policy_impact = 0.0
        for policy in climate_policies:
            effectiveness = policy.get("effectiveness", self.policy_effectiveness)
            coverage = policy.get("coverage", 1.0)
            policy_impact -= effectiveness * coverage * self.base_warming_rate * 0.5
        
        # 随机波动
        random_factor = random_state.normal(1.0, self.volatility)
        
        # 计算总变化
        total_change = (base_change + emission_impact + policy_impact) * random_factor
        
        # 应用气候敏感性
        total_change *= self.climate_sensitivity
        
        # 计算新温度
        new_temp = current_temp + total_change
        
        # 限制在合理范围内
        new_temp = max(self.min_global_temp, min(self.max_global_temp, new_temp))
        
        return new_temp
    
    def _evolve_regional_temperatures(self, regional_temps: Dict[str, float],
                                     global_temp: float,
                                     random_state: np.random.RandomState) -> Dict[str, float]:
        """
        演化地区温度
        
        Args:
            regional_temps: 当前地区温度变化
            global_temp: 全球温度变化
            random_state: 随机数生成器
            
        Returns:
            新的地区温度变化
        """
        updated_regional_temps = {}
        
        for region in self.regions:
            # 获取当前地区温度
            current_temp = regional_temps.get(region, 0.0)
            
            # 地区放大效应（某些地区变暖更快）
            region_factor = 1.0 + (self.region_vulnerability.get(region, 0.5) * 0.5)
            
            # 随机地区波动
            region_volatility = random_state.normal(0.0, self.volatility * 0.5)
            
            # 向全球温度靠拢，但保留地区特性
            convergence_factor = 0.3  # 收敛因子
            new_temp = current_temp * (1 - convergence_factor) + global_temp * region_factor * convergence_factor + region_volatility
            
            updated_regional_temps[region] = new_temp
        
        return updated_regional_temps
    
    def _update_greenhouse_gases(self, current_concentrations: Dict[str, float],
                                emissions: Dict[str, float]) -> Dict[str, float]:
        """
        更新温室气体浓度
        
        Args:
            current_concentrations: 当前温室气体浓度
            emissions: 温室气体排放
            
        Returns:
            更新后的温室气体浓度
        """
        updated_concentrations = current_concentrations.copy()
        
        # 默认背景清除率（简化模型）
        natural_removal_rate = 0.02  # 每年2%的自然清除
        
        for gas in self.greenhouse_gases:
            current = updated_concentrations.get(gas, 0.0)
            
            # 添加新排放
            new_emissions = emissions.get(gas, 0.0)
            
            # 应用自然清除
            natural_removal = current * natural_removal_rate
            
            # 更新浓度
            updated_concentrations[gas] = max(0, current + new_emissions - natural_removal)
        
        return updated_concentrations
    
    def _generate_extreme_events(self, regional_temps: Dict[str, float],
                               climate_policies: List[Dict[str, Any]],
                               random_state: np.random.RandomState) -> List[Dict[str, Any]]:
        """
        生成极端天气事件
        
        Args:
            regional_temps: 地区温度
            climate_policies: 气候政策
            random_state: 随机数生成器
            
        Returns:
            极端事件列表
        """
        extreme_events = []
        
        # 政策可以略微降低极端事件概率
        policy_reduction = 0.0
        for policy in climate_policies:
            policy_reduction += policy.get("effectiveness", self.policy_effectiveness) * 0.01
        
        # 温度上升增加极端事件概率
        temp_amplification = 1.0
        avg_temp_change = sum(regional_temps.values()) / len(regional_temps)
        if avg_temp_change > 1.0:
            temp_amplification = 1.0 + (avg_temp_change - 1.0) * 0.5
        
        # 为每个地区生成可能的极端事件
        for region in self.regions:
            # 地区脆弱性影响事件概率
            vulnerability = self.region_vulnerability.get(region, 0.5)
            
            # 计算最终事件概率
            event_probability = min(0.5, self.extreme_event_probability * vulnerability * temp_amplification * (1 - policy_reduction))
            
            # 检查是否发生事件
            if random_state.random() < event_probability:
                # 选择事件类型
                event_type = random_state.choice(self.extreme_event_types)
                
                # 事件严重性基于温度和随机因素
                severity = min(10.0, 3.0 + (regional_temps.get(region, 0.0) * 2.0) + random_state.random() * 5.0)
                
                # 经济影响基于严重性和地区
                economic_impact = severity * 10 * (0.5 + vulnerability)
                
                event = {
                    "type": event_type,
                    "region": region,
                    "severity": severity,
                    "economic_impact_billions_usd": economic_impact,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "description": f"在{region}发生{event_type}，严重性为{severity:.1f}级"
                }
                
                extreme_events.append(event)
                self.logger.info(f"极端气候事件: {event['description']}")
        
        return extreme_events
    
    def _calculate_sea_level_rise(self, global_temp: float) -> float:
        """
        计算海平面上升
        
        Args:
            global_temp: 全球温度变化
            
        Returns:
            海平面上升（米）
        """
        # 简化的海平面上升模型
        if global_temp <= 0:
            return 0
        
        # 温度与海平面上升的非线性关系
        sea_level_rise = 0.2 * (1 - np.exp(-global_temp * 1.5))
        
        return sea_level_rise
    
    def _calculate_climate_impacts(self, regional_temps: Dict[str, float],
                                 extreme_events: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        计算气候影响
        
        Args:
            regional_temps: 地区温度
            extreme_events: 极端事件
            
        Returns:
            地区气候影响
        """
        impacts = {}
        
        for region in self.regions:
            impacts[region] = {
                "economic": 0.0,
                "agricultural": 0.0,
                "health": 0.0,
                "ecosystem": 0.0,
                "total": 0.0
            }
            
            # 温度影响
            temp = regional_temps.get(region, 0.0)
            
            # 温度对各领域的影响
            if temp > 0:
                # 经济影响
                impacts[region]["economic"] = temp * 5.0 * self.region_vulnerability.get(region, 0.5)
                
                # 农业影响
                if 0 < temp < 2.0:
                    # 适度升温可能有正面影响
                    impacts[region]["agricultural"] = -temp * 2.0
                else:
                    # 过热负面影响
                    impacts[region]["agricultural"] = temp * 8.0 * self.region_vulnerability.get(region, 0.5)
                
                # 健康影响
                impacts[region]["health"] = temp * 3.0 * self.region_vulnerability.get(region, 0.5)
                
                # 生态系统影响
                impacts[region]["ecosystem"] = temp * 10.0 * self.region_vulnerability.get(region, 0.5)
            
            # 极端事件影响
            for event in extreme_events:
                if event["region"] == region:
                    event_impact = event["severity"] * 2.0
                    impacts[region]["economic"] += event_impact
                    impacts[region]["agricultural"] += event_impact * 0.5
                    impacts[region]["health"] += event_impact * 0.8
                    impacts[region]["ecosystem"] += event_impact * 1.2
            
            # 计算总影响
            impacts[region]["total"] = (
                impacts[region]["economic"] +
                impacts[region]["agricultural"] +
                impacts[region]["health"] +
                impacts[region]["ecosystem"]
            )
        
        return impacts
    
    def calculate_co2_equivalent(self, emissions: Dict[str, float]) -> float:
        """
        计算CO2当量排放
        
        Args:
            emissions: 各温室气体排放
            
        Returns:
            CO2当量排放
        """
        co2_eq = 0.0
        
        for gas, amount in emissions.items():
            if gas in self.co2_equivalence:
                co2_eq += amount * self.co2_equivalence[gas]
        
        return co2_eq
    
    def get_climate_risk_index(self, climate_state: Dict[str, Any], region: str = None) -> float:
        """
        计算气候风险指数
        
        Args:
            climate_state: 气候状态
            region: 地区（可选，不提供则计算全球）
            
        Returns:
            风险指数（0-10）
        """
        if region and region in self.regions:
            # 计算特定地区风险
            temp = climate_state["regional_temperatures"].get(region, 0.0)
            vulnerability = self.region_vulnerability.get(region, 0.5)
            
            # 计算该地区极端事件数量和严重性
            regional_events = [e for e in climate_state.get("recent_extreme_events", []) if e["region"] == region]
            event_risk = sum(e["severity"] for e in regional_events) * 0.1
            
            # 计算风险指数
            risk_index = min(10.0, temp * vulnerability * 2.0 + event_risk)
        else:
            # 计算全球风险
            global_temp = climate_state["global_temperature"]
            sea_level = climate_state["sea_level_rise"]
            total_events = len(climate_state.get("recent_extreme_events", []))
            
            # 计算全球风险指数
            risk_index = min(10.0, global_temp * 3.0 + sea_level * 20.0 + total_events * 0.5)
        
        return risk_index

# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 初始化气候模型
    climate_model = ClimateModel()
    
    # 创建初始气候状态
    initial_climate = {
        "global_temperature": 1.1,  # 当前全球温度上升1.1°C
        "regional_temperatures": {
            "North America": 1.2,
            "South America": 1.0,
            "Europe": 1.3,
            "Africa": 1.5,
            "Asia": 1.4,
            "Oceania": 1.2,
            "Antarctica": 2.0
        },
        "greenhouse_gas_concentrations": {
            "CO2": 415,  # ppm
            "CH4": 1.87,  # ppm
            "N2O": 0.332,  # ppm
            "F-gases": 0.0009  # ppm
        },
        "sea_level_rise": 0.25,  # 米
        "recent_extreme_events": [],
        "climate_impacts": {},
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # 人类活动数据
    human_activities = {
        "carbon_emissions": {
            "China": 10065,  # 百万吨
            "USA": 5416,
            "India": 2654,
            "Russia": 1711,
            "Japan": 1162
        },
        "greenhouse_gas_emissions": {
            "CO2": 36.3,  # 十亿吨/年
            "CH4": 0.35,  # 十亿吨/年
            "N2O": 0.08,  # 十亿吨/年
            "F-gases": 0.02  # 十亿吨/年
        },
        "climate_policies": [
            {
                "name": "Paris Agreement",
                "effectiveness": 0.3,
                "coverage": 0.7
            },
            {
                "name": "Renewable Energy Transition",
                "effectiveness": 0.4,
                "coverage": 0.5
            }
        ]
    }
    
    print("初始气候状态:")
    print(f"全球温度上升: {initial_climate['global_temperature']}°C")
    print(f"海平面上升: {initial_climate['sea_level_rise']}米")
    
    # 模拟5个时间步
    random_state = np.random.RandomState(42)
    for step in range(5):
        new_climate = climate_model.evolve(initial_climate, human_activities, random_state)
        initial_climate = new_climate
        
        print(f"\n时间步 {step+1}:")
        print(f"全球温度上升: {initial_climate['global_temperature']:.2f}°C")
        print(f"海平面上升: {initial_climate['sea_level_rise']:.2f}米")
        print(f"极端事件数: {len(initial_climate['recent_extreme_events'])}")
        
        # 显示风险指数
        print(f"全球气候风险指数: {climate_model.get_climate_risk_index(initial_climate):.2f}")
        for region in ["Africa", "Asia"]:
            print(f"{region}气候风险指数: {climate_model.get_climate_risk_index(initial_climate, region):.2f}")
    
    # 计算CO2当量
    co2_eq = climate_model.calculate_co2_equivalent(human_activities["greenhouse_gas_emissions"])
    print(f"\n总CO2当量排放: {co2_eq:.2f} 十亿吨/年")