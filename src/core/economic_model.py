#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
经济模型模块
负责模拟全球经济系统的演化
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random

from ..utils.logger import get_logger

class EconomicModel:
    """经济模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化经济模型
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("EconomicModel")
        self.config = config
        
        # 模型参数
        self.volatility = config.get("economic_volatility", 0.02)  # 经济波动度
        self.gdp_growth_rate = config.get("baseline_gdp_growth_rate", 0.03)  # 基础GDP增长率
        self.inflation_target = config.get("inflation_target", 0.02)  # 通胀目标
        self.unemployment_target = config.get("unemployment_target", 0.05)  # 失业率目标
        self.debt_threshold = config.get("debt_threshold", 1.0)  # 债务GDP比阈值
        
        # 国家参数映射
        self.country_parameters = config.get("country_parameters", {})
        
        # 领域间影响权重
        self.cross_impact_weights = {
            "political": config.get("political_impact_weight", 0.3),
            "technological": config.get("technological_impact_weight", 0.4),
            "climate": config.get("climate_impact_weight", 0.2)
        }
        
        self.logger.info("经济模型初始化完成")
    
    def evolve(self, economic_state: Dict[str, Any], 
              current_time: datetime, 
              random_state: np.random.RandomState) -> Dict[str, Any]:
        """
        演化经济状态
        
        Args:
            economic_state: 当前经济状态
            current_time: 当前时间
            random_state: 随机数生成器
            
        Returns:
            更新后的经济状态
        """
        # 复制状态以避免修改原始数据
        new_state = economic_state.copy()
        
        # 演化GDP
        if "gdp" in new_state:
            new_state["gdp"] = self._evolve_gdp(new_state["gdp"], random_state)
        
        # 演化通胀
        if "inflation" in new_state:
            new_state["inflation"] = self._evolve_inflation(
                new_state["inflation"], 
                new_state.get("gdp", {}),
                random_state
            )
        
        # 演化失业率
        if "unemployment" in new_state:
            new_state["unemployment"] = self._evolve_unemployment(
                new_state["unemployment"],
                new_state.get("gdp", {}),
                random_state
            )
        
        # 演化债务
        if "debt" in new_state and "gdp" in new_state:
            new_state["debt"] = self._evolve_debt(
                new_state["debt"],
                new_state["gdp"],
                new_state.get("inflation", {}),
                random_state
            )
        
        # 演化贸易
        if "trade" in new_state:
            new_state["trade"] = self._evolve_trade(new_state["trade"], random_state)
        
        # 演化市场情绪
        if "market_sentiment" in new_state:
            new_state["market_sentiment"] = self._evolve_market_sentiment(
                new_state["market_sentiment"],
                new_state.get("gdp", {}),
                new_state.get("inflation", {}),
                random_state
            )
        
        # 更新时间戳
        new_state["last_updated"] = current_time.isoformat()
        
        return new_state
    
    def _evolve_gdp(self, gdp_data: Dict[str, float], 
                   random_state: np.random.RandomState) -> Dict[str, float]:
        """
        演化GDP数据
        
        Args:
            gdp_data: 国家GDP数据
            random_state: 随机数生成器
            
        Returns:
            更新后的GDP数据
        """
        updated_gdp = {}
        
        for country, gdp in gdp_data.items():
            # 获取国家特定参数
            country_params = self.country_parameters.get(country, {})
            growth_rate = country_params.get("gdp_growth_rate", self.gdp_growth_rate)
            country_volatility = country_params.get("volatility", self.volatility)
            
            # 添加随机波动
            random_component = random_state.normal(0, country_volatility)
            
            # 计算新的GDP
            new_gdp = gdp * (1 + growth_rate + random_component)
            
            # 确保GDP不会变为负数
            new_gdp = max(0.1, new_gdp)
            
            updated_gdp[country] = new_gdp
        
        return updated_gdp
    
    def _evolve_inflation(self, inflation_data: Dict[str, float],
                         gdp_data: Dict[str, float],
                         random_state: np.random.RandomState) -> Dict[str, float]:
        """
        演化通胀数据
        
        Args:
            inflation_data: 国家通胀数据
            gdp_data: 国家GDP数据
            random_state: 随机数生成器
            
        Returns:
            更新后的通胀数据
        """
        updated_inflation = {}
        
        for country, inflation in inflation_data.items():
            # 获取国家特定参数
            country_params = self.country_parameters.get(country, {})
            target = country_params.get("inflation_target", self.inflation_target)
            adjustment_speed = country_params.get("inflation_adjustment_speed", 0.2)
            
            # GDP增长对通胀的影响（快速增长可能导致通胀上升）
            gdp_impact = 0.0
            if country in gdp_data:
                # 假设GDP增长率对通胀有正相关影响
                gdp_change = random_state.normal(0.03, 0.02)  # 模拟GDP增长率
                gdp_impact = 0.5 * gdp_change  # GDP增长50%转化为通胀
            
            # 均值回归：趋向目标通胀率
            mean_reversion = adjustment_speed * (target - inflation)
            
            # 随机冲击
            random_shock = random_state.normal(0, 0.5)  # 通胀随机波动
            
            # 计算新的通胀率
            new_inflation = inflation + mean_reversion + gdp_impact + random_shock / 100
            
            # 限制范围
            new_inflation = max(-5.0, min(10.0, new_inflation))  # -5%到10%
            
            updated_inflation[country] = new_inflation
        
        return updated_inflation
    
    def _evolve_unemployment(self, unemployment_data: Dict[str, float],
                           gdp_data: Dict[str, float],
                           random_state: np.random.RandomState) -> Dict[str, float]:
        """
        演化失业率数据
        
        Args:
            unemployment_data: 国家失业率数据
            gdp_data: 国家GDP数据
            random_state: 随机数生成器
            
        Returns:
            更新后的失业率数据
        """
        updated_unemployment = {}
        
        for country, unemployment in unemployment_data.items():
            # 获取国家特定参数
            country_params = self.country_parameters.get(country, {})
            target = country_params.get("unemployment_target", self.unemployment_target)
            adjustment_speed = country_params.get("unemployment_adjustment_speed", 0.15)
            
            # GDP增长对失业率的影响（奥肯定律）
            # 通常认为GDP增长每超过自然增长率1%，失业率下降约0.5%
            gdp_impact = 0.0
            if country in gdp_data:
                # 模拟GDP增长率与自然增长率的差异
                growth_gap = random_state.normal(0, 0.02)
                gdp_impact = -0.5 * growth_gap * 100  # 转化为失业率变化百分比
            
            # 均值回归：趋向目标失业率
            mean_reversion = adjustment_speed * (target - unemployment)
            
            # 随机冲击
            random_shock = random_state.normal(0, 0.2)  # 失业率随机波动
            
            # 计算新的失业率
            new_unemployment = unemployment + mean_reversion + gdp_impact + random_shock
            
            # 限制范围
            new_unemployment = max(0.1, min(30.0, new_unemployment))  # 0.1%到30%
            
            updated_unemployment[country] = new_unemployment
        
        return updated_unemployment
    
    def _evolve_debt(self, debt_data: Dict[str, float],
                    gdp_data: Dict[str, float],
                    inflation_data: Dict[str, float],
                    random_state: np.random.RandomState) -> Dict[str, float]:
        """
        演化债务数据
        
        Args:
            debt_data: 国家债务数据
            gdp_data: 国家GDP数据
            inflation_data: 国家通胀数据
            random_state: 随机数生成器
            
        Returns:
            更新后的债务数据
        """
        updated_debt = {}
        
        for country, debt in debt_data.items():
            # 基础债务增长率
            base_debt_growth = random_state.normal(0.02, 0.01)  # 2%的基础债务增长
            
            # 通胀对债务的实际价值影响（通胀会降低实际债务负担）
            inflation_impact = 0.0
            if country in inflation_data:
                inflation_impact = -inflation_data[country] / 100
            
            # 债务GDP比率
            debt_gdp_ratio = 0.0
            if country in gdp_data and gdp_data[country] > 0:
                debt_gdp_ratio = debt / gdp_data[country]
            
            # 当债务GDP比率超过阈值时，可能会减少新增债务
            debt_constraint = 0.0
            if debt_gdp_ratio > self.debt_threshold:
                # 债务过高时减少债务增长
                excess_debt = (debt_gdp_ratio - self.debt_threshold)
                debt_constraint = -0.5 * excess_debt  # 债务过高会降低债务增长率
            
            # 计算新的债务
            new_debt = debt * (1 + base_debt_growth + inflation_impact + debt_constraint)
            
            # 确保债务不会变为负数
            new_debt = max(0.0, new_debt)
            
            updated_debt[country] = new_debt
        
        return updated_debt
    
    def _evolve_trade(self, trade_data: Dict[str, Any],
                     random_state: np.random.RandomState) -> Dict[str, Any]:
        """
        演化贸易数据
        
        Args:
            trade_data: 贸易数据
            random_state: 随机数生成器
            
        Returns:
            更新后的贸易数据
        """
        updated_trade = trade_data.copy()
        
        # 演化贸易流量
        if "flows" in updated_trade:
            updated_flows = {}
            
            for (source, destination), flow in updated_trade["flows"].items():
                # 贸易流量随机波动
                random_factor = random_state.normal(1.0, 0.05)  # 5%的波动
                new_flow = flow * random_factor
                
                # 确保贸易流量不会变为负数
                new_flow = max(0.0, new_flow)
                
                updated_flows[(source, destination)] = new_flow
            
            updated_trade["flows"] = updated_flows
        
        # 演化贸易余额
        if "balances" in updated_trade:
            updated_balances = {}
            
            for country, balance in updated_trade["balances"].items():
                # 贸易余额随机波动
                random_change = random_state.normal(0, 10)  # 假设贸易余额以十亿美元为单位
                new_balance = balance + random_change
                
                updated_balances[country] = new_balance
            
            updated_trade["balances"] = updated_balances
        
        return updated_trade
    
    def _evolve_market_sentiment(self, sentiment_data: Dict[str, float],
                                gdp_data: Dict[str, float],
                                inflation_data: Dict[str, float],
                                random_state: np.random.RandomState) -> Dict[str, float]:
        """
        演化市场情绪
        
        Args:
            sentiment_data: 市场情绪数据（-100到100）
            gdp_data: GDP数据
            inflation_data: 通胀数据
            random_state: 随机数生成器
            
        Returns:
            更新后的市场情绪
        """
        updated_sentiment = {}
        
        for country, sentiment in sentiment_data.items():
            # GDP增长对情绪的影响
            gdp_impact = 0.0
            if country in gdp_data:
                # 假设GDP增长对情绪有正面影响
                gdp_growth = random_state.normal(0.03, 0.02)
                gdp_impact = 20.0 * gdp_growth  # GDP增长3%可能带来0.6点情绪提升
            
            # 通胀对情绪的影响（高通胀通常负面影响情绪）
            inflation_impact = 0.0
            if country in inflation_data:
                inflation_impact = -2.0 * inflation_data[country]  # 通胀率1%可能导致情绪下降0.02
            
            # 均值回归：市场情绪趋向中性（0）
            mean_reversion = 0.1 * (0 - sentiment)  # 10%的速度回归中性
            
            # 随机冲击
            random_shock = random_state.normal(0, 5.0)  # 市场情绪随机波动
            
            # 计算新的市场情绪
            new_sentiment = sentiment + gdp_impact + inflation_impact + mean_reversion + random_shock
            
            # 限制在-100到100范围内
            new_sentiment = max(-100, min(100, new_sentiment))
            
            updated_sentiment[country] = new_sentiment
        
        return updated_sentiment
    
    def calculate_economic_health_index(self, economic_state: Dict[str, Any]) -> Dict[str, float]:
        """
        计算经济健康指数
        
        Args:
            economic_state: 经济状态
            
        Returns:
            各国家的经济健康指数（0-100）
        """
        health_indices = {}
        
        # 获取所有需要的国家列表
        countries = set()
        if "gdp" in economic_state:
            countries.update(economic_state["gdp"].keys())
        if "inflation" in economic_state:
            countries.update(economic_state["inflation"].keys())
        if "unemployment" in economic_state:
            countries.update(economic_state["unemployment"].keys())
        
        for country in countries:
            # 初始化健康指数
            health_index = 50.0
            
            # GDP评分（占30%）
            if "gdp" in economic_state and country in economic_state["gdp"]:
                gdp_value = economic_state["gdp"][country]
                # 假设GDP越高，评分越高（简化处理）
                gdp_score = min(100, gdp_value * 2)  # 50万亿美元对应满分
                health_index += (gdp_score - 50) * 0.3
            
            # 通胀评分（占25%）
            if "inflation" in economic_state and country in economic_state["inflation"]:
                inflation_value = economic_state["inflation"][country]
                # 通胀越接近目标，评分越高
                target = self.country_parameters.get(country, {}).get("inflation_target", self.inflation_target)
                inflation_deviation = abs(inflation_value - target)
                inflation_score = max(0, 100 - inflation_deviation * 20)  # 偏差0.5%为满分
                health_index += (inflation_score - 50) * 0.25
            
            # 失业率评分（占25%）
            if "unemployment" in economic_state and country in economic_state["unemployment"]:
                unemployment_value = economic_state["unemployment"][country]
                # 失业率越低，评分越高
                unemployment_score = max(0, 100 - unemployment_value * 10)  # 5%失业率为满分
                health_index += (unemployment_score - 50) * 0.25
            
            # 债务GDP比评分（占20%）
            if "debt" in economic_state and "gdp" in economic_state and \
               country in economic_state["debt"] and country in economic_state["gdp"]:
                gdp = economic_state["gdp"][country]
                if gdp > 0:
                    debt_gdp_ratio = economic_state["debt"][country] / gdp
                    # 债务GDP比越低，评分越高
                    debt_score = max(0, 100 - debt_gdp_ratio * 50)  # 200%债务GDP比为0分
                    health_index += (debt_score - 50) * 0.2
            
            # 限制在0-100范围内
            health_indices[country] = max(0, min(100, health_index))
        
        return health_indices

# 测试代码
if __name__ == "__main__":
    # 简单的测试配置
    config = {
        "economic_volatility": 0.02,
        "baseline_gdp_growth_rate": 0.03,
        "inflation_target": 0.02,
        "unemployment_target": 0.05,
        "debt_threshold": 1.0,
        "country_parameters": {
            "US": {
                "gdp_growth_rate": 0.025,
                "volatility": 0.015,
                "inflation_target": 0.02,
                "unemployment_target": 0.04
            },
            "China": {
                "gdp_growth_rate": 0.05,
                "volatility": 0.025,
                "inflation_target": 0.03,
                "unemployment_target": 0.05
            }
        }
    }
    
    # 创建经济模型
    eco_model = EconomicModel(config)
    
    # 创建初始经济状态
    initial_state = {
        "gdp": {
            "US": 25.0,  # 万亿美元
            "China": 18.0,
            "Japan": 4.0,
            "Germany": 3.5,
            "India": 3.0
        },
        "inflation": {
            "US": 3.0,  # 百分比
            "EU": 4.0,
            "China": 2.5,
            "Japan": 0.5
        },
        "unemployment": {
            "US": 4.0,  # 百分比
            "EU": 7.0,
            "China": 5.5,
            "Japan": 2.5
        },
        "debt": {
            "US": 30.0,  # 万亿美元
            "Japan": 12.0,
            "EU": 15.0,
            "China": 10.0
        },
        "trade": {
            "flows": {
                ("China", "US"): 500.0,  # 十亿美元
                ("US", "China"): 150.0,
                ("Germany", "EU"): 300.0,
                ("Japan", "US"): 150.0
            },
            "balances": {
                "US": -1000.0,
                "China": 800.0,
                "Germany": 300.0,
                "Japan": 100.0
            }
        },
        "market_sentiment": {
            "US": 50.0,
            "EU": 40.0,
            "China": 60.0,
            "Japan": 45.0
        }
    }
    
    # 模拟演化
    current_time = datetime.now()
    random_state = np.random.RandomState(42)
    
    print("初始经济状态:")
    print(f"GDP: {initial_state['gdp']}")
    print(f"通胀: {initial_state['inflation']}")
    print(f"失业率: {initial_state['unemployment']}")
    print()
    
    # 演化10步
    for step in range(10):
        next_time = current_time + timedelta(days=30)  # 假设每月一步
        new_state = eco_model.evolve(initial_state, next_time, random_state)
        initial_state = new_state
    
    print("演化后经济状态:")
    print(f"GDP: {initial_state['gdp']}")
    print(f"通胀: {initial_state['inflation']}")
    print(f"失业率: {initial_state['unemployment']}")
    print()
    
    # 计算经济健康指数
    health_indices = eco_model.calculate_economic_health_index(initial_state)
    print("经济健康指数:")
    for country, index in health_indices.items():
        print(f"{country}: {index:.2f}")