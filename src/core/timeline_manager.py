#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
时间线管理模块
负责管理多个世界线的演化、分叉和评估
"""

import logging
import uuid
import copy
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import random

from ..utils.logger import get_logger
from ..utils.random_utils import get_seeded_random
from ..models.evaluator import WorldLineEvaluator
from ..data.data_types import WorldLine
from .world_simulator import WorldSimulator


class TimelineManager:
    """时间线管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化时间线管理器
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("TimelineManager")
        self.config = config
        
        # 世界线存储
        self.worldlines: Dict[str, WorldLine] = {}
        
        # 最大世界线数量限制
        self.max_worldlines = config.get("max_worldlines", 1000)
        
        # 保留世界线的数量（每次修剪时）
        self.keep_worldlines = config.get("keep_worldlines", 500)
        
        # 分叉配置
        self.fork_probability = config.get("fork_probability", 0.3)
        self.max_forks_per_step = config.get("max_forks_per_step", 2)
        
        # 初始化世界模拟器和评估器
        self.world_simulator = WorldSimulator(config)
        self.evaluator = WorldLineEvaluator(config)
        
        self.logger.info("时间线管理器初始化完成")
    
    def create_initial_worldline(self, initial_state: Dict[str, Any], 
                               seed: Optional[int] = None) -> WorldLine:
        """
        创建初始世界线
        
        Args:
            initial_state: 初始状态
            seed: 随机种子（可选）
            
        Returns:
            创建的世界线
        """
        now = datetime.now()
        worldline = {
            "id": str(uuid.uuid4()),
            "seed": seed if seed is not None else random.randint(1, 1000000),
            "generation": 0,
            "birth_time": now.isoformat(),
            "parent_ids": [],
            "survival_score": 0.0,
            "current_time": now.isoformat(),
            "state": initial_state,
            "events": [],
            "history": []
        }
        
        # 评估初始世界线
        worldline["survival_score"] = self.evaluator.evaluate(worldline)
        
        # 添加到存储
        self.worldlines[worldline["id"]] = worldline
        
        self.logger.info(f"创建初始世界线: {worldline['id']}")
        return worldline
    
    def evolve_all(self, steps: int = 1) -> List[WorldLine]:
        """
        演化所有活跃的世界线
        
        Args:
            steps: 演化步数
            
        Returns:
            演化后的世界线列表
        """
        if not self.worldlines:
            self.logger.warning("没有世界线可供演化")
            return []
        
        self.logger.info(f"演化所有 {len(self.worldlines)} 个世界线，每线 {steps} 步")
        
        # 获取所有世界线
        worldline_list = list(self.worldlines.values())
        
        # 演化所有世界线
        evolved_worldlines = self.world_simulator.simulate(worldline_list, steps)
        
        # 更新存储并评估演化后的世界线
        for worldline in evolved_worldlines:
            # 评估并更新生存分数
            worldline["survival_score"] = self.evaluator.evaluate(worldline)
            self.worldlines[worldline["id"]] = worldline
            
            # 尝试分叉
            self._attempt_fork(worldline)
        
        # 检查是否需要修剪世界线
        self._prune_worldlines()
        
        return list(self.worldlines.values())
    
    def _attempt_fork(self, worldline: WorldLine) -> None:
        """
        智能尝试分叉世界线（优化版本）
        
        Args:
            worldline: 要尝试分叉的世界线
        """
        # 只有高价值的世界线才进行分叉
        survival_score = worldline.get("survival_score", 0.0)
        
        # 根据生存分数动态调整分叉概率
        dynamic_fork_prob = min(0.5, self.fork_probability * (1 + survival_score))
        
        rng = get_seeded_random(worldline["seed"] + len(worldline["history"]))
        
        # 决定是否分叉
        if rng.random() > dynamic_fork_prob:
            return
        
        # 根据世界线价值决定分叉数量
        if survival_score > 0.7:  # 高价值世界线
            max_possible_forks = min(4, self.max_forks_per_step)
        elif survival_score > 0.5:  # 中等价值世界线
            max_possible_forks = min(2, self.max_forks_per_step)
        else:  # 低价值世界线
            max_possible_forks = 1
        
        num_forks = rng.randint(1, max_possible_forks)
        remaining_slots = self.max_worldlines - len(self.worldlines)
        actual_forks = min(num_forks, remaining_slots)
        
        if actual_forks == 0:
            self.logger.warning("已达到最大世界线数量，停止分叉")
            return
        
        for _ in range(actual_forks):
            # 创建分叉世界线
            fork_worldline = self._create_fork(worldline, rng)
            
            # 添加到存储
            self.worldlines[fork_worldline["id"]] = fork_worldline
            
            self.logger.info(f"创建世界线分叉: {fork_worldline['id']} 来自 {worldline['id']}")
    
    def _create_fork(self, parent_worldline: WorldLine, rng) -> WorldLine:
        """
        从父世界线创建分叉
        
        Args:
            parent_worldline: 父世界线
            rng: 随机数生成器
            
        Returns:
            创建的分叉世界线
        """
        # 复制父世界线
        fork_worldline = copy.deepcopy(parent_worldline)
        
        # 生成新ID和种子
        fork_worldline["id"] = str(uuid.uuid4())
        fork_worldline["seed"] = rng.randint(1, 1000000)
        fork_worldline["generation"] = parent_worldline["generation"] + 1
        fork_worldline["birth_time"] = datetime.now().isoformat()
        fork_worldline["parent_ids"] = parent_worldline["parent_ids"] + [parent_worldline["id"]]
        
        # 引入随机变化到状态
        self._introduce_random_changes(fork_worldline["state"], rng)
        
        # 生成一个分叉事件
        fork_event = {
            "type": "timeline_fork",
            "severity": "minor",
            "description": f"世界线分叉自 {parent_worldline['id']}",
            "timestamp": fork_worldline["current_time"],
            "event_id": str(uuid.uuid4()),
            "processed": False
        }
        fork_worldline["events"].append(fork_event)
        
        # 评估分叉世界线
        fork_worldline["survival_score"] = self.evaluator.evaluate(fork_worldline)
        
        return fork_worldline
    
    def _introduce_random_changes(self, state: Dict[str, Any], rng) -> None:
        """
        向世界线状态引入随机变化
        
        Args:
            state: 世界线状态
            rng: 随机数生成器
        """
        # 经济状态随机变化
        if "economic" in state:
            eco_state = state["economic"]
            # GDP变化 ±0~5%
            if "gdp" in eco_state:
                for country in eco_state["gdp"]:
                    change_factor = rng.uniform(0.95, 1.05)
                    eco_state["gdp"][country] *= change_factor
            
            # 通胀变化 ±0~1%
            if "inflation" in eco_state:
                for country in eco_state["inflation"]:
                    change = rng.uniform(-1.0, 1.0)
                    eco_state["inflation"][country] = max(0, eco_state["inflation"][country] + change)
        
        # 政治状态随机变化
        if "political" in state and "tensions" in state["political"]:
            pol_state = state["political"]
            # 紧张度变化 ±0~10%
            for region in pol_state["tensions"]:
                change = rng.uniform(-10, 10)
                pol_state["tensions"][region] = max(0, min(100, pol_state["tensions"][region] + change))
        
        # 技术状态随机变化
        if "technological" in state:
            tech_state = state["technological"]
            # 随机选择一个技术领域进行小幅突破
            tech_areas = [key for key in tech_state if isinstance(tech_state[key], (int, float))]
            if tech_areas:
                target_area = rng.choice(tech_areas)
                change = rng.uniform(1, 5)
                tech_state[target_area] = min(100, tech_state[target_area] + change)
    
    def _prune_worldlines(self) -> None:
        """
        修剪世界线，保留最有价值的世界线
        """
        if len(self.worldlines) <= self.keep_worldlines:
            return
        
        self.logger.info(f"修剪世界线: 从 {len(self.worldlines)} 减少到 {self.keep_worldlines}")
        
        # 按生存分数排序
        sorted_worldlines = sorted(
            self.worldlines.values(), 
            key=lambda w: w["survival_score"], 
            reverse=True
        )
        
        # 保留前N个世界线
        keep_ids = {w["id"] for w in sorted_worldlines[:self.keep_worldlines]}
        
        # 删除其他世界线
        to_delete = [w_id for w_id in self.worldlines if w_id not in keep_ids]
        for w_id in to_delete:
            del self.worldlines[w_id]
        
        self.logger.info(f"修剪完成，删除了 {len(to_delete)} 个世界线")
    
    def get_worldline(self, worldline_id: str) -> Optional[WorldLine]:
        """
        获取指定ID的世界线
        
        Args:
            worldline_id: 世界线ID
            
        Returns:
            世界线，如果不存在则返回None
        """
        return self.worldlines.get(worldline_id)
    
    def get_all_worldlines(self) -> List[WorldLine]:
        """
        获取所有世界线
        
        Returns:
            世界线列表
        """
        return list(self.worldlines.values())
    
    def get_top_worldlines(self, count: int = 10) -> List[WorldLine]:
        """
        获取评分最高的世界线
        
        Args:
            count: 要返回的世界线数量
            
        Returns:
            按生存分数排序的世界线列表
        """
        sorted_worldlines = sorted(
            self.worldlines.values(), 
            key=lambda w: w["survival_score"], 
            reverse=True
        )
        return sorted_worldlines[:count]
    
    def export_worldline(self, worldline_id: str) -> Dict[str, Any]:
        """
        导出世界线数据
        
        Args:
            worldline_id: 世界线ID
            
        Returns:
            世界线数据字典
        """
        worldline = self.get_worldline(worldline_id)
        if not worldline:
            raise ValueError(f"世界线 {worldline_id} 不存在")
        
        # 返回深拷贝以避免修改原始数据
        return copy.deepcopy(worldline)
    
    def shutdown(self) -> None:
        """
        关闭时间线管理器，释放资源
        """
        self.logger.info("关闭世界模拟器...")
        self.world_simulator.shutdown()
        
        self.logger.info("时间线管理器已关闭")


# 测试代码
if __name__ == "__main__":
    # 简单的测试配置
    config = {
        "max_worldlines": 100,
        "keep_worldlines": 50,
        "fork_probability": 0.3,
        "max_forks_per_step": 2,
        "parallel_workers": 4,
        "max_simulation_depth": 90
    }
    
    # 创建时间线管理器
    manager = TimelineManager(config)
    
    # 创建初始状态
    initial_state = {
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
            "ai_progress": 70,
            "renewable_energy": 45,
            "quantum_computing": 30
        },
        "climate": {
            "global_temp_increase": 1.2,
            "co2_levels": 419,
            "extreme_events": 42
        }
    }
    
    # 创建初始世界线
    print("创建初始世界线...")
    manager.create_initial_worldline(initial_state, seed=42)
    
    # 演化几次
    print("开始演化...")
    for i in range(3):
        print(f"\n演化步骤 {i+1}:")
        worldlines = manager.evolve_all(steps=1)
        print(f"当前世界线数量: {len(worldlines)}")
        
        # 打印评分最高的世界线
        top_worldline = manager.get_top_worldlines(1)[0]
        print(f"最高评分世界线: {top_worldline['id']}，评分: {top_worldline['survival_score']:.4f}")
    
    # 关闭管理器
    print("\n关闭管理器...")
    manager.shutdown()