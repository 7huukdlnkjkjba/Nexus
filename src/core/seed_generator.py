#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
种子生成器模块
负责生成初始世界线种子和基于幸存者的新一代世界线
"""

import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid

from ..utils.random_utils import get_seeded_random
from ..utils.logger import get_logger
from ..data.data_types import WorldLine, RealityData

class SeedGenerator:
    """世界线种子生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化种子生成器
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("SeedGenerator")
        self.config = config
        
        # 扰动参数
        self.economic_perturbation_range = config.get("economic_perturbation_range", 0.05)  # 经济参数5%的扰动范围
        self.political_perturbation_range = config.get("political_perturbation_range", 0.1)   # 政治参数10%的扰动范围
        self.technological_perturbation_range = config.get("technological_perturbation_range", 0.15)  # 技术参数15%的扰动范围
        self.climate_perturbation_range = config.get("climate_perturbation_range", 0.03)  # 气候参数3%的扰动范围
        
        # 事件参数
        self.major_event_probability = config.get("major_event_probability", 0.05)  # 5%概率发生重大事件
        self.medium_event_probability = config.get("medium_event_probability", 0.2)  # 20%概率发生中等事件
        self.minor_event_probability = config.get("minor_event_probability", 0.4)  # 40%概率发生小事件
        
        # 继承参数
        self.survivor_inheritance_rate = config.get("survivor_inheritance_rate", 0.7)  # 70%的参数从幸存者继承
        
        self.logger.info("种子生成器初始化完成")
    
    def generate_initial_seeds(self, reality_data: RealityData, count: Optional[int] = None) -> List[WorldLine]:
        """
        生成初始世界线种子
        
        Args:
            reality_data: 现实数据
            count: 生成的世界线数量，默认为配置中的num_worldlines
            
        Returns:
            世界线列表
        """
        num_worldlines = count or self.config.get("num_worldlines", 100000)
        self.logger.info(f"生成 {num_worldlines} 个初始世界线种子")
        
        worldlines = []
        
        for i in range(num_worldlines):
            # 为每个世界线生成唯一ID和种子
            worldline_id = str(uuid.uuid4())
            random_seed = random.randint(1, 2**32 - 1)
            rng = get_seeded_random(random_seed)
            
            # 基于现实数据创建带扰动的初始状态
            initial_state = self._create_perturbed_state(reality_data, rng)
            
            # 可能添加随机事件
            events = self._generate_random_events(rng)
            
            # 创建世界线
            worldline = {
                "id": worldline_id,
                "seed": random_seed,
                "generation": 0,
                "birth_time": datetime.now().isoformat(),
                "parent_ids": [],  # 初始世界线没有父世界线
                "survival_score": 0.0,
                "current_time": datetime.now().isoformat(),
                "state": initial_state,
                "events": events,
                "history": []  # 将记录世界线的演化历史
            }
            
            worldlines.append(worldline)
            
            # 记录进度
            if (i + 1) % 10000 == 0:
                self.logger.info(f"已生成 {i + 1} 个初始世界线种子")
        
        self.logger.info(f"初始世界线种子生成完成，共 {len(worldlines)} 个")
        return worldlines
    
    def generate_new_generation(self, survivors: List[WorldLine], 
                              reality_data: RealityData, 
                              count: Optional[int] = None) -> List[WorldLine]:
        """
        基于存活的世界线生成新一代世界线
        
        Args:
            survivors: 存活的世界线列表
            reality_data: 当前的现实数据
            count: 生成的世界线数量，默认为配置中的num_worldlines
            
        Returns:
            新一代世界线列表
        """
        if not survivors:
            self.logger.warning("没有存活的世界线，回退到生成初始种子")
            return self.generate_initial_seeds(reality_data, count)
        
        num_worldlines = count or self.config.get("num_worldlines", 100000)
        self.logger.info(f"基于 {len(survivors)} 个存活世界线生成 {num_worldlines} 个新一代世界线")
        
        worldlines = []
        
        # 根据生存分数加权选择父世界线
        survivor_scores = [w.get("survival_score", 1.0) for w in survivors]
        total_score = sum(survivor_scores)
        
        if total_score == 0:
            # 如果所有分数都为0，使用均匀分布
            weights = None
        else:
            weights = [score / total_score for score in survivor_scores]
        
        for i in range(num_worldlines):
            # 为每个世界线生成唯一ID和种子
            worldline_id = str(uuid.uuid4())
            random_seed = random.randint(1, 2**32 - 1)
            rng = get_seeded_random(random_seed)
            
            # 选择父世界线
            parent_worldline = rng.choices(survivors, weights=weights)[0]
            
            # 创建混合状态：部分继承父世界线，部分使用现实数据
            mixed_state = self._create_mixed_state(
                parent_worldline["state"], 
                reality_data,
                rng
            )
            
            # 可能添加随机事件
            events = self._generate_random_events(rng)
            
            # 创建世界线
            worldline = {
                "id": worldline_id,
                "seed": random_seed,
                "generation": parent_worldline["generation"] + 1,
                "birth_time": datetime.now().isoformat(),
                "parent_ids": [parent_worldline["id"]],
                "survival_score": 0.0,
                "current_time": datetime.now().isoformat(),
                "state": mixed_state,
                "events": events,
                "history": []
            }
            
            worldlines.append(worldline)
            
            # 记录进度
            if (i + 1) % 10000 == 0:
                self.logger.info(f"已生成 {i + 1} 个新一代世界线")
        
        self.logger.info(f"新一代世界线生成完成，共 {len(worldlines)} 个")
        return worldlines
    
    def _create_perturbed_state(self, reality_data: RealityData, rng) -> Dict[str, Any]:
        """创建带有随机扰动的初始状态"""
        perturbed_state = {}
        
        # 经济数据扰动
        if "economic" in reality_data:
            perturbed_state["economic"] = self._perturb_section(
                reality_data["economic"], 
                self.economic_perturbation_range, 
                rng
            )
        
        # 政治数据扰动
        if "political" in reality_data:
            perturbed_state["political"] = self._perturb_section(
                reality_data["political"], 
                self.political_perturbation_range, 
                rng
            )
        
        # 技术数据扰动
        if "technological" in reality_data:
            perturbed_state["technological"] = self._perturb_section(
                reality_data["technological"], 
                self.technological_perturbation_range, 
                rng
            )
        
        # 气候数据扰动
        if "climate" in reality_data:
            perturbed_state["climate"] = self._perturb_section(
                reality_data["climate"], 
                self.climate_perturbation_range, 
                rng
            )
        
        # 复制其他不变的数据
        for key, value in reality_data.items():
            if key not in perturbed_state:
                perturbed_state[key] = value
        
        return perturbed_state
    
    def _create_mixed_state(self, parent_state: Dict[str, Any], 
                           reality_data: RealityData, 
                           rng) -> Dict[str, Any]:
        """创建混合状态：部分继承父世界线，部分使用现实数据"""
        mixed_state = {}
        
        # 遍历父世界线的所有部分
        for section_key, section_data in parent_state.items():
            if rng.random() < self.survivor_inheritance_rate:
                # 继承父世界线的数据，但添加小的扰动
                perturbation_factor = 0.5  # 子世界线的扰动比初始世界线小
                mixed_state[section_key] = self._perturb_section(
                    section_data,
                    self._get_perturbation_range(section_key) * perturbation_factor,
                    rng
                )
            else:
                # 使用最新的现实数据
                if section_key in reality_data:
                    mixed_state[section_key] = self._perturb_section(
                        reality_data[section_key],
                        self._get_perturbation_range(section_key),
                        rng
                    )
                else:
                    # 如果现实数据中没有，就使用父世界线的数据
                    mixed_state[section_key] = section_data
        
        # 添加现实数据中可能有的新部分
        for section_key, section_data in reality_data.items():
            if section_key not in mixed_state:
                mixed_state[section_key] = self._perturb_section(
                    section_data,
                    self._get_perturbation_range(section_key),
                    rng
                )
        
        return mixed_state
    
    def _get_perturbation_range(self, section_key: str) -> float:
        """根据部分类型获取扰动范围"""
        if "economic" in section_key.lower():
            return self.economic_perturbation_range
        elif "political" in section_key.lower():
            return self.political_perturbation_range
        elif "technological" in section_key.lower():
            return self.technological_perturbation_range
        elif "climate" in section_key.lower():
            return self.climate_perturbation_range
        else:
            return 0.05  # 默认5%
    
    def _perturb_section(self, data: Any, perturbation_range: float, rng) -> Any:
        """扰动数据部分"""
        if isinstance(data, dict):
            # 递归扰动字典
            perturbed = {}
            for key, value in data.items():
                perturbed[key] = self._perturb_section(value, perturbation_range, rng)
            return perturbed
        elif isinstance(data, list):
            # 递归扰动列表
            perturbed = []
            for item in data:
                perturbed.append(self._perturb_section(item, perturbation_range, rng))
            return perturbed
        elif isinstance(data, (int, float)):
            # 对数值添加随机扰动
            perturbation = rng.uniform(-perturbation_range, perturbation_range)
            return data * (1 + perturbation)
        elif isinstance(data, bool):
            # 对布尔值有小概率翻转
            if rng.random() < perturbation_range:
                return not data
            return data
        else:
            # 其他类型保持不变
            return data
    
    def _generate_random_events(self, rng) -> List[Dict[str, Any]]:
        """生成随机事件"""
        events = []
        
        # 可能的重大事件
        major_events = [
            {"type": "economic_crisis", "severity": "major", "description": "全球性经济危机爆发"},
            {"type": "conflict", "severity": "major", "description": "主要地区发生武装冲突"},
            {"type": "tech_breakthrough", "severity": "major", "description": "能源存储技术重大突破"},
            {"type": "natural_disaster", "severity": "major", "description": "超大规模自然灾害"},
            {"type": "pandemic", "severity": "major", "description": "新型传染病全球大流行"}
        ]
        
        # 可能的中等事件
        medium_events = [
            {"type": "policy_change", "severity": "medium", "description": "主要经济体政策大幅调整"},
            {"type": "trade_dispute", "severity": "medium", "description": "主要贸易伙伴间贸易争端升级"},
            {"type": "tech_adoption", "severity": "medium", "description": "人工智能技术广泛应用"},
            {"type": "climate_event", "severity": "medium", "description": "极端气候事件频发"},
            {"type": "market_crash", "severity": "medium", "description": "主要金融市场大幅下跌"}
        ]
        
        # 可能的小事件
        minor_events = [
            {"type": "minor_policy", "severity": "minor", "description": "小规模政策调整"},
            {"type": "minor_conflict", "severity": "minor", "description": "局部地区小规模冲突"},
            {"type": "minor_tech", "severity": "minor", "description": "特定领域技术进展"},
            {"type": "minor_market", "severity": "minor", "description": "市场短期波动"},
            {"type": "minor_climate", "severity": "minor", "description": "地区性气候异常"}
        ]
        
        # 生成重大事件
        if rng.random() < self.major_event_probability:
            event = rng.choice(major_events)
            event["timestamp"] = datetime.now().isoformat()
            event["event_id"] = str(uuid.uuid4())
            events.append(event)
        
        # 生成中等事件
        if rng.random() < self.medium_event_probability:
            event = rng.choice(medium_events)
            event["timestamp"] = datetime.now().isoformat()
            event["event_id"] = str(uuid.uuid4())
            events.append(event)
        
        # 生成小事件
        if rng.random() < self.minor_event_probability:
            event = rng.choice(minor_events)
            event["timestamp"] = datetime.now().isoformat()
            event["event_id"] = str(uuid.uuid4())
            events.append(event)
        
        return events

# 测试代码
if __name__ == "__main__":
    # 简单的测试配置
    config = {
        "economic_perturbation_range": 0.05,
        "political_perturbation_range": 0.1,
        "technological_perturbation_range": 0.15,
        "climate_perturbation_range": 0.03,
        "major_event_probability": 0.05,
        "medium_event_probability": 0.2,
        "minor_event_probability": 0.4,
        "survivor_inheritance_rate": 0.7
    }
    
    # 模拟现实数据
    reality_data = {
        "economic": {
            "gdp": {"US": 25.46, "China": 17.96, "Japan": 4.23},
            "inflation": {"US": 3.7, "EU": 4.3, "UK": 6.7},
            "unemployment": {"US": 3.8, "EU": 6.1, "Japan": 2.5}
        },
        "political": {
            "alliances": ["NATO", "EU", "ASEAN"],
            "tensions": {"Taiwan_Strait": 65, "Ukraine": 80, "Middle_East": 75}
        },
        "technological": {
            "ai_progress": 70,  # 0-100
            "renewable_energy": 45,
            "quantum_computing": 30
        },
        "climate": {
            "global_temp_increase": 1.2,
            "co2_levels": 419,  # ppm
            "extreme_events": 42  # 年发生次数
        }
    }
    
    generator = SeedGenerator(config)
    
    # 测试生成初始种子
    initial_worldlines = generator.generate_initial_seeds(reality_data, count=10)
    print(f"生成了 {len(initial_worldlines)} 个初始世界线")
    print("第一个世界线示例:", initial_worldlines[0]["state"]["economic"]["gdp"])
    
    # 模拟一些存活的世界线
    survivors = initial_worldlines[:3]  # 模拟前3个存活
    for i, w in enumerate(survivors):
        w["survival_score"] = 10.0 - i  # 给予不同的生存分数
    
    # 测试生成新一代
    new_generation = generator.generate_new_generation(survivors, reality_data, count=5)
    print(f"生成了 {len(new_generation)} 个新一代世界线")
    print("第一个新一代世界线父ID:", new_generation[0]["parent_ids"])