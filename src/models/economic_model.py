#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
经济模型模块
负责模拟全球经济系统的演化
"""

import logging
import random
from typing import Dict, Any, Optional
import math
from datetime import datetime

from ..utils.logger import get_logger
from ..utils.random_utils import get_seeded_random


class EconomicModel:
    """
    经济模型
    模拟全球经济系统的演化
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化经济模型
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("EconomicModel")
        self.config = config
        
        # 经济参数配置
        self.gdp_growth_params = config.get("gdp_growth", {
            "baseline_rate": 0.025,  # 基准增长率（2.5%）
            "volatility": 0.01,       # 波动范围
            "technology_factor": 0.3, # 技术对GDP的影响因子
            "climate_impact": 0.2     # 气候对GDP的影响因子
        })
        
        self.inflation_params = config.get("inflation", {
            "target_rate": 0.02,      # 目标通胀率（2%）
            "volatility": 0.01,       # 波动范围
            "growth_sensitivity": 0.5 # 对经济增长的敏感度
        })
        
        self.unemployment_params = config.get("unemployment", {
            "natural_rate": 0.05,     # 自然失业率（5%）
            "volatility": 0.01,       # 波动范围
            "growth_sensitivity": -0.4 # 对经济增长的敏感度（负相关）
        })
        
        # 国家间经济联系强度
        self.trade_connections = config.get("trade_connections", {
            "US": {"China": 0.15, "EU": 0.2, "Japan": 0.1, "UK": 0.08},
            "China": {"US": 0.15, "EU": 0.12, "Japan": 0.07, "UK": 0.05},
            "EU": {"US": 0.2, "China": 0.12, "Japan": 0.06, "UK": 0.15},
            "Japan": {"US": 0.1, "China": 0.07, "EU": 0.06, "UK": 0.03},
            "UK": {"US": 0.08, "China": 0.05, "EU": 0.15, "Japan": 0.03}
        })
        
        # 主要经济体初始参数
        self.economic_powers = config.get("economic_powers", [
            "US", "China", "EU", "Japan", "UK", "India", "Germany", 
            "France", "Brazil", "Italy", "Canada", "South_Korea", 
            "Russia", "Australia", "Spain"
        ])
        
        self.logger.info("经济模型初始化完成")
    
    def evolve(self, economic_state: Dict[str, Any], current_time: datetime, 
              rng: random.Random) -> Dict[str, Any]:
        """
        演化经济状态
        
        Args:
            economic_state: 当前经济状态
            current_time: 当前时间
            rng: 随机数生成器
            
        Returns:
            演化后的经济状态
        """
        try:
            # 创建新状态以避免修改原始数据
            new_state = economic_state.copy()
            
            # 确保所有必要的子状态存在
            new_state["gdp"] = new_state.get("gdp", {})
            new_state["inflation"] = new_state.get("inflation", {})
            new_state["unemployment"] = new_state.get("unemployment", {})
            new_state["trade"] = new_state.get("trade", {})
            
            # 生成各经济体的经济冲击
            economic_shocks = self._generate_economic_shocks(new_state, rng)
            
            # 计算GDP增长
            new_state["gdp"] = self._update_gdp(new_state["gdp"], economic_shocks, rng)
            
            # 根据GDP增长更新通胀率
            new_state["inflation"] = self._update_inflation(
                new_state["inflation"], new_state["gdp"], economic_shocks, rng
            )
            
            # 根据GDP增长更新失业率
            new_state["unemployment"] = self._update_unemployment(
                new_state["unemployment"], new_state["gdp"], economic_shocks, rng
            )
            
            # 更新贸易数据
            new_state["trade"] = self._update_trade(new_state["gdp"], rng)
            
            # 更新经济周期状态
            new_state["economic_cycle"] = self._update_economic_cycle(new_state, rng)
            
            return new_state
            
        except Exception as e:
            self.logger.error(f"演化经济状态失败: {str(e)}")
            return economic_state  # 失败时返回原始状态
    
    def _generate_economic_shocks(self, economic_state: Dict[str, Any], 
                                 rng: random.Random) -> Dict[str, float]:
        """
        生成随机经济冲击
        
        Args:
            economic_state: 当前经济状态
            rng: 随机数生成器
            
        Returns:
            各经济体的经济冲击字典
        """
        shocks = {}
        
        # 为每个经济体生成冲击
        for country in self.economic_powers:
            # 1%的概率发生重大冲击
            if rng.random() < 0.01:
                # 重大冲击：-5% 到 +5%
                shock = rng.uniform(-0.05, 0.05)
                shocks[country] = shock
                self.logger.debug(f"对 {country} 的重大经济冲击: {shock:.2%}")
            # 10%的概率发生中等冲击
            elif rng.random() < 0.1:
                # 中等冲击：-2% 到 +2%
                shock = rng.uniform(-0.02, 0.02)
                shocks[country] = shock
            else:
                # 轻微冲击：-0.5% 到 +0.5%
                shock = rng.uniform(-0.005, 0.005)
                shocks[country] = shock
        
        return shocks
    
    def _update_gdp(self, gdp_data: Dict[str, float], 
                   shocks: Dict[str, float], rng: random.Random) -> Dict[str, float]:
        """
        更新GDP数据
        
        Args:
            gdp_data: 当前GDP数据
            shocks: 经济冲击
            rng: 随机数生成器
            
        Returns:
            更新后的GDP数据
        """
        new_gdp = gdp_data.copy()
        
        # 确保所有主要经济体都有GDP数据
        for country in self.economic_powers:
            if country not in new_gdp:
                # 设置默认GDP值（万亿美元）
                default_gdp = self._get_default_gdp(country)
                new_gdp[country] = default_gdp
        
        # 计算GDP增长率并更新
        for country, current_gdp in new_gdp.items():
            # 获取基础增长率
            base_growth = self.gdp_growth_params["baseline_rate"]
            
            # 添加随机波动
            volatility = self.gdp_growth_params["volatility"]
            random_factor = rng.uniform(-volatility, volatility)
            
            # 添加经济冲击
            shock = shocks.get(country, 0)
            
            # 计算贸易伙伴影响
            trade_impact = self._calculate_trade_impact(country, new_gdp, rng)
            
            # 总增长率
            total_growth = base_growth + random_factor + shock + trade_impact
            
            # 确保增长率在合理范围内（-10%到15%）
            total_growth = max(-0.1, min(0.15, total_growth))
            
            # 更新GDP
            new_gdp[country] = current_gdp * (1 + total_growth)
        
        return new_gdp
    
    def _get_default_gdp(self, country: str) -> float:
        """
        获取国家默认GDP值（万亿美元）
        
        Args:
            country: 国家名称
            
        Returns:
            默认GDP值
        """
        default_gdps = {
            "US": 25.46,
            "China": 17.96,
            "EU": 16.64,  # 欧盟整体
            "Japan": 4.23,
            "UK": 3.33,
            "India": 3.39,
            "Germany": 4.07,
            "France": 2.94,
            "Brazil": 1.92,
            "Italy": 2.01,
            "Canada": 2.14,
            "South_Korea": 1.79,
            "Russia": 2.21,
            "Australia": 1.61,
            "Spain": 1.41
        }
        
        return default_gdps.get(country, 0.5)  # 默认0.5万亿美元
    
    def _calculate_trade_impact(self, country: str, gdp_data: Dict[str, float], 
                               rng: random.Random) -> float:
        """
        计算贸易伙伴对GDP的影响
        
        Args:
            country: 目标国家
            gdp_data: GDP数据
            rng: 随机数生成器
            
        Returns:
            贸易影响值
        """
        impact = 0.0
        
        # 获取该国家的贸易联系
        country_connections = self.trade_connections.get(country, {})
        
        for partner, connection_strength in country_connections.items():
            if partner in gdp_data:
                # 模拟贸易伙伴GDP波动的影响
                # 简化模型：贸易伙伴GDP增长的一部分会传导到本国
                partner_gdp = gdp_data[partner]
                # 这里假设我们知道前一期的GDP，简化处理
                # 实际应该从历史数据中获取
                
                # 随机模拟伙伴国经济状态对本国的影响
                partner_impact = connection_strength * rng.uniform(-0.01, 0.01)
                impact += partner_impact
        
        return impact
    
    def _update_inflation(self, inflation_data: Dict[str, float], 
                         gdp_data: Dict[str, float], 
                         shocks: Dict[str, float], rng: random.Random) -> Dict[str, float]:
        """
        更新通胀率
        
        Args:
            inflation_data: 当前通胀率数据
            gdp_data: GDP数据
            shocks: 经济冲击
            rng: 随机数生成器
            
        Returns:
            更新后的通胀率数据
        """
        new_inflation = inflation_data.copy()
        target_rate = self.inflation_params["target_rate"]
        volatility = self.inflation_params["volatility"]
        growth_sensitivity = self.inflation_params["growth_sensitivity"]
        
        # 确保所有主要经济体都有通胀数据
        for country in self.economic_powers:
            if country not in new_inflation:
                # 设置默认通胀率（百分比）
                new_inflation[country] = target_rate * 100
        
        # 更新每个国家的通胀率
        for country, current_inflation in new_inflation.items():
            # 将通胀率从百分比转换为小数
            current_inflation_decimal = current_inflation / 100
            
            # 基础变化：向目标率靠拢
            target_adjustment = (target_rate - current_inflation_decimal) * 0.3
            
            # 随机波动
            random_factor = rng.uniform(-volatility, volatility)
            
            # GDP增长影响（简化：假设GDP增长与通胀正相关）
            gdp_impact = growth_sensitivity * rng.uniform(0.005, 0.015)
            
            # 经济冲击影响
            shock_impact = shocks.get(country, 0) * 0.2
            
            # 总变化
            total_change = target_adjustment + random_factor + gdp_impact + shock_impact
            
            # 更新通胀率
            new_inflation_decimal = current_inflation_decimal + total_change
            
            # 确保通胀率在合理范围内（-5%到20%）
            new_inflation_decimal = max(-0.05, min(0.2, new_inflation_decimal))
            
            # 转换回百分比
            new_inflation[country] = new_inflation_decimal * 100
        
        return new_inflation
    
    def _update_unemployment(self, unemployment_data: Dict[str, float], 
                            gdp_data: Dict[str, float], 
                            shocks: Dict[str, float], rng: random.Random) -> Dict[str, float]:
        """
        更新失业率
        
        Args:
            unemployment_data: 当前失业率数据
            gdp_data: GDP数据
            shocks: 经济冲击
            rng: 随机数生成器
            
        Returns:
            更新后的失业率数据
        """
        new_unemployment = unemployment_data.copy()
        natural_rate = self.unemployment_params["natural_rate"]
        volatility = self.unemployment_params["volatility"]
        growth_sensitivity = self.unemployment_params["growth_sensitivity"]
        
        # 确保所有主要经济体都有失业率数据
        for country in self.economic_powers:
            if country not in new_unemployment:
                # 设置默认失业率（百分比）
                new_unemployment[country] = natural_rate * 100
        
        # 更新每个国家的失业率
        for country, current_unemployment in new_unemployment.items():
            # 将失业率从百分比转换为小数
            current_unemployment_decimal = current_unemployment / 100
            
            # 基础变化：向自然失业率靠拢
            natural_adjustment = (natural_rate - current_unemployment_decimal) * 0.2
            
            # 随机波动
            random_factor = rng.uniform(-volatility, volatility)
            
            # GDP增长影响（奥肯定律：GDP增长与失业率负相关）
            gdp_impact = growth_sensitivity * rng.uniform(0.01, 0.02)
            
            # 经济冲击影响（通常是正相关）
            shock_impact = shocks.get(country, 0) * -0.3  # 负冲击增加失业率
            
            # 总变化
            total_change = natural_adjustment + random_factor + gdp_impact + shock_impact
            
            # 更新失业率
            new_unemployment_decimal = current_unemployment_decimal + total_change
            
            # 确保失业率在合理范围内（0%到30%）
            new_unemployment_decimal = max(0, min(0.3, new_unemployment_decimal))
            
            # 转换回百分比
            new_unemployment[country] = new_unemployment_decimal * 100
        
        return new_unemployment
    
    def _update_trade(self, gdp_data: Dict[str, float], rng: random.Random) -> Dict[str, Any]:
        """
        更新贸易数据
        
        Args:
            gdp_data: GDP数据
            rng: 随机数生成器
            
        Returns:
            贸易数据
        """
        trade_data = {}
        
        # 计算各国之间的贸易流量（简化模型）
        for exporter in self.economic_powers:
            if exporter in gdp_data:
                trade_data[exporter] = {}
                
                # 出口额通常是GDP的一定比例
                export_ratio = rng.uniform(0.15, 0.3)  # 15%-30%的GDP用于出口
                total_exports = gdp_data[exporter] * export_ratio
                
                # 分配出口到各个贸易伙伴
                partner_shares = {}
                total_share = 0
                
                # 使用贸易联系强度作为分配基础
                for importer in self.economic_powers:
                    if importer != exporter and importer in self.trade_connections.get(exporter, {}):
                        base_share = self.trade_connections[exporter].get(importer, 0)
                        # 添加随机变化
                        adjusted_share = base_share * rng.uniform(0.8, 1.2)
                        partner_shares[importer] = adjusted_share
                        total_share += adjusted_share
                
                # 归一化并分配出口额
                if total_share > 0:
                    for importer, share in partner_shares.items():
                        normalized_share = share / total_share
                        trade_data[exporter][importer] = total_exports * normalized_share
        
        return trade_data
    
    def _update_economic_cycle(self, economic_state: Dict[str, Any], 
                              rng: random.Random) -> str:
        """
        更新经济周期状态
        
        Args:
            economic_state: 经济状态
            rng: 随机数生成器
            
        Returns:
            经济周期状态描述
        """
        # 基于主要经济体的平均GDP增长率判断经济周期
        gdp_data = economic_state.get("gdp", {})
        
        if not gdp_data:
            return "stable"
        
        # 计算主要经济体的平均GDP增长率（简化：比较当前与假设的上期）
        # 实际应该从历史数据中计算
        growth_rates = []
        major_economies = ["US", "China", "EU", "Japan", "UK"]
        
        for country in major_economies:
            if country in gdp_data:
                # 随机模拟增长率（实际应计算真实增长）
                growth_rate = rng.uniform(-0.02, 0.06)
                growth_rates.append(growth_rate)
        
        if not growth_rates:
            return "stable"
        
        avg_growth_rate = sum(growth_rates) / len(growth_rates)
        
        # 判断经济周期
        if avg_growth_rate < -0.02:
            return "recession"
        elif avg_growth_rate < 0:
            return "contraction"
        elif avg_growth_rate < 0.02:
            return "stable"
        elif avg_growth_rate < 0.04:
            return "expansion"
        else:
            return "boom"
    
    def calculate_economic_health(self, economic_state: Dict[str, Any]) -> float:
        """
        计算整体经济健康度
        
        Args:
            economic_state: 经济状态
            
        Returns:
            0-1之间的健康度分数
        """
        # 检查必要数据
        if not economic_state or "gdp" not in economic_state:
            return 0.5
        
        gdp_data = economic_state["gdp"]
        inflation_data = economic_state.get("inflation", {})
        unemployment_data = economic_state.get("unemployment", {})
        
        scores = []
        
        # 计算主要经济体的经济健康度
        major_economies = ["US", "China", "EU", "Japan", "UK"]
        
        for country in major_economies:
            if country in gdp_data:
                country_score = 1.0
                
                # GDP评分（简化：基于GDP规模）
                gdp = gdp_data[country]
                gdp_score = min(gdp / 30.0, 1.0)  # 30万亿美元为满分
                country_score *= gdp_score
                
                # 通胀率评分
                if country in inflation_data:
                    inflation = inflation_data[country]
                    if 1 <= inflation <= 3:
                        inflation_score = 1.0  # 理想区间
                    elif 0 <= inflation < 1 or 3 < inflation <= 5:
                        inflation_score = 0.8
                    elif inflation < 0 or 5 < inflation <= 10:
                        inflation_score = 0.5
                    else:
                        inflation_score = 0.2
                    country_score *= inflation_score
                
                # 失业率评分
                if country in unemployment_data:
                    unemployment = unemployment_data[country]
                    if unemployment <= 4:
                        unemployment_score = 1.0  # 理想区间
                    elif unemployment <= 6:
                        unemployment_score = 0.8
                    elif unemployment <= 10:
                        unemployment_score = 0.5
                    else:
                        unemployment_score = 0.2
                    country_score *= unemployment_score
                
                scores.append(country_score)
        
        # 计算平均健康度
        if scores:
            avg_score = sum(scores) / len(scores)
            return avg_score
        else:
            return 0.5


# 测试代码
if __name__ == "__main__":
    # 简单的测试配置
    config = {
        "gdp_growth": {
            "baseline_rate": 0.025,
            "volatility": 0.01
        },
        "inflation": {
            "target_rate": 0.02,
            "volatility": 0.01
        },
        "unemployment": {
            "natural_rate": 0.05,
            "volatility": 0.01
        }
    }
    
    # 创建经济模型
    model = EconomicModel(config)
    
    # 初始经济状态
    initial_state = {
        "gdp": {
            "US": 25.46,
            "China": 17.96,
            "EU": 16.64,
            "Japan": 4.23,
            "UK": 3.33
        },
        "inflation": {
            "US": 3.7,
            "EU": 4.3,
            "UK": 6.7,
            "Japan": 1.2,
            "China": 2.1
        },
        "unemployment": {
            "US": 3.8,
            "EU": 6.1,
            "Japan": 2.5,
            "UK": 4.2,
            "China": 5.5
        },
        "trade": {}
    }
    
    # 模拟演化
    from datetime import datetime
    from ..utils.random_utils import get_seeded_random
    
    rng = get_seeded_random(42)
    print("初始经济状态:")
    print(f"GDP: {initial_state['gdp']}")
    print(f"通胀率: {initial_state['inflation']}")
    print(f"失业率: {initial_state['unemployment']}")
    
    # 演化一次
    new_state = model.evolve(initial_state, datetime.now(), rng)
    
    print("\n演化后的经济状态:")
    print(f"GDP: {new_state['gdp']}")
    print(f"通胀率: {new_state['inflation']}")
    print(f"失业率: {new_state['unemployment']}")
    print(f"经济周期: {new_state.get('economic_cycle')}")
    
    # 计算经济健康度
    health_score = model.calculate_economic_health(new_state)
    print(f"\n经济健康度: {health_score:.4f}")