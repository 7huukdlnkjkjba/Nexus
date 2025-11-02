#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
世界模拟器模块
负责模拟世界线的演化过程
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import concurrent.futures
import multiprocessing

from ..utils.logger import get_logger
from ..utils.random_utils import get_seeded_random
from ..models.economic_model import EconomicModel
from ..models.political_model import PoliticalModel
from ..models.technological_model import TechnologicalModel
from ..models.climate_model import ClimateModel
from ..models.interaction_model import InteractionModel
from ..data.data_types import WorldLine

class WorldSimulator:
    """世界线模拟器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化世界模拟器
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("WorldSimulator")
        self.config = config
        
        # 并行计算设置
        self.max_workers = config.get("parallel_workers", min(16, multiprocessing.cpu_count() * 2))
        self.chunk_size = config.get("simulation_chunk_size", 100)
        
        # 模拟器精度和深度
        self.max_simulation_depth = config.get("max_simulation_depth", 90)  # 最多模拟90天
        
        # 初始化各领域模型
        self.logger.info("初始化领域模型...")
        self.economic_model = EconomicModel(config)
        self.political_model = PoliticalModel(config)
        self.technological_model = TechnologicalModel(config)
        self.climate_model = ClimateModel(config)
        self.interaction_model = InteractionModel(config)
        
        # 初始化线程池
        self.executor = None
        self._initialize_executor()
        
        self.logger.info("世界模拟器初始化完成")
    
    def _initialize_executor(self):
        """初始化线程池执行器"""
        if self.executor:
            self.executor.shutdown(wait=False)
        
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="WorldSimulator"
        )
    
    def shutdown(self):
        """关闭模拟器，释放资源"""
        if self.executor:
            self.logger.info("关闭线程池执行器...")
            self.executor.shutdown(wait=True)
            self.executor = None
        
        self.logger.info("世界模拟器已关闭")
    
    def simulate(self, worldlines: List[WorldLine], steps: int = 1) -> List[WorldLine]:
        """
        模拟世界线演化
        
        Args:
            worldlines: 要模拟的世界线列表
            steps: 模拟的步数（天）
            
        Returns:
            模拟后的世界线列表
        """
        if not worldlines:
            self.logger.warning("没有世界线需要模拟")
            return []
        
        self.logger.info(f"开始模拟 {len(worldlines)} 个世界线，每线 {steps} 步")
        start_time = time.time()
        
        # 将世界线分成多个批次进行并行处理
        simulated_worldlines = []
        
        # 如果世界线数量很少，直接串行处理
        if len(worldlines) <= self.chunk_size:
            for worldline in worldlines:
                try:
                    simulated = self._simulate_single_worldline(worldline, steps)
                    simulated_worldlines.append(simulated)
                except Exception as e:
                    self.logger.error(f"模拟世界线 {worldline.get('id')} 失败: {str(e)}")
        else:
            # 并行处理
            futures = []
            for i in range(0, len(worldlines), self.chunk_size):
                chunk = worldlines[i:i + self.chunk_size]
                future = self.executor.submit(self._simulate_chunk, chunk, steps)
                futures.append(future)
            
            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_results = future.result()
                    simulated_worldlines.extend(chunk_results)
                except Exception as e:
                    self.logger.error(f"处理模拟批次失败: {str(e)}")
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"模拟完成，耗时 {elapsed_time:.2f} 秒，平均每世界线 {elapsed_time/len(worldlines):.4f} 秒")
        
        return simulated_worldlines
    
    def _simulate_chunk(self, worldlines: List[WorldLine], steps: int) -> List[WorldLine]:
        """
        模拟一个批次的世界线
        
        Args:
            worldlines: 世界线批次
            steps: 模拟步数
            
        Returns:
            模拟后的世界线列表
        """
        results = []
        for worldline in worldlines:
            try:
                simulated = self._simulate_single_worldline(worldline, steps)
                results.append(simulated)
            except Exception as e:
                self.logger.error(f"模拟世界线 {worldline.get('id')} 失败: {str(e)}")
        return results
    
    def _simulate_single_worldline(self, worldline: WorldLine, steps: int) -> WorldLine:
        """
        模拟单个世界线的演化
        
        Args:
            worldline: 要模拟的世界线
            steps: 模拟的步数
            
        Returns:
            模拟后的世界线
        """
        # 复制世界线以避免修改原始数据
        simulated_worldline = worldline.copy()
        
        # 获取随机数生成器（基于世界线种子确保可重现性）
        rng = get_seeded_random(simulated_worldline["seed"])
        
        # 获取当前状态
        state = simulated_worldline["state"].copy()
        
        # 获取当前时间
        current_time = datetime.fromisoformat(simulated_worldline["current_time"])
        
        # 模拟每一步
        for step in range(steps):
            # 记录当前状态作为历史
            history_entry = {
                "timestamp": current_time.isoformat(),
                "state": state.copy()
            }
            simulated_worldline["history"].append(history_entry)
            
            # 检查是否已经达到最大模拟深度
            days_since_birth = (current_time - datetime.fromisoformat(simulated_worldline["birth_time"])).days
            if days_since_birth >= self.max_simulation_depth:
                self.logger.debug(f"世界线 {simulated_worldline['id']} 已达到最大模拟深度")
                break
            
            # 1. 应用领域模型进行演化
            # 经济演化
            state["economic"] = self.economic_model.evolve(
                state["economic"], 
                current_time, 
                rng
            )
            
            # 政治演化
            state["political"] = self.political_model.evolve(
                state["political"], 
                current_time, 
                rng
            )
            
            # 技术演化
            state["technological"] = self.technological_model.evolve(
                state["technological"], 
                current_time, 
                rng
            )
            
            # 气候演化
            state["climate"] = self.climate_model.evolve(
                state["climate"], 
                current_time, 
                rng
            )
            
            # 2. 处理领域间的相互影响
            state = self.interaction_model.apply_interactions(
                state,
                current_time,
                rng
            )
            
            # 3. 检查并处理事件
            state = self._process_events(
                state, 
                simulated_worldline["events"], 
                current_time,
                rng
            )
            
            # 4. 可能产生新的随机事件
            if rng.random() < 0.1:  # 10%的概率产生新事件
                new_event = self._generate_random_event(rng, current_time)
                simulated_worldline["events"].append(new_event)
            
            # 更新时间
            current_time += timedelta(days=1)
        
        # 更新世界线状态
        simulated_worldline["current_time"] = current_time.isoformat()
        simulated_worldline["state"] = state
        
        return simulated_worldline
    
    def _process_events(self, state: Dict[str, Any], events: List[Dict[str, Any]],
                      current_time: datetime, rng) -> Dict[str, Any]:
        """
        处理世界线中的事件对状态的影响
        
        Args:
            state: 当前状态
            events: 事件列表
            current_time: 当前时间
            rng: 随机数生成器
            
        Returns:
            更新后的状态
        """
        # 深拷贝状态以避免修改原始数据
        updated_state = state.copy()
        
        # 处理每个事件
        for event in events:
            # 检查事件是否已经处理过
            if "processed" in event and event["processed"]:
                continue
            
            # 检查事件时间是否应该发生
            event_time = datetime.fromisoformat(event["timestamp"])
            if (current_time - event_time).days >= 0:
                # 处理事件
                updated_state = self._apply_event_effects(updated_state, event, rng)
                event["processed"] = True
                event["processing_time"] = current_time.isoformat()
        
        return updated_state
    
    def _apply_event_effects(self, state: Dict[str, Any], event: Dict[str, Any],
                           rng) -> Dict[str, Any]:
        """
        应用事件对状态的影响
        
        Args:
            state: 当前状态
            event: 要应用的事件
            rng: 随机数生成器
            
        Returns:
            更新后的状态
        """
        event_type = event.get("type", "")
        severity = event.get("severity", "minor")
        
        # 根据事件类型和严重程度应用影响
        if event_type == "economic_crisis":
            # 经济危机影响
            if severity == "major":
                # 主要经济体GDP下降10-20%
                for country in state["economic"]["gdp"]:
                    state["economic"]["gdp"][country] *= rng.uniform(0.8, 0.9)
                # 失业率上升5-10%
                for country in state["economic"]["unemployment"]:
                    state["economic"]["unemployment"][country] += rng.uniform(5, 10)
            elif severity == "medium":
                # 主要经济体GDP下降5-10%
                for country in state["economic"]["gdp"]:
                    state["economic"]["gdp"][country] *= rng.uniform(0.9, 0.95)
                # 失业率上升2-5%
                for country in state["economic"]["unemployment"]:
                    state["economic"]["unemployment"][country] += rng.uniform(2, 5)
        
        elif event_type == "conflict":
            # 冲突影响
            if severity == "major":
                # 增加政治紧张度
                for region in state["political"]["tensions"]:
                    state["political"]["tensions"][region] = min(100, state["political"]["tensions"][region] + rng.uniform(20, 40))
                # 影响经济
                for country in state["economic"]["gdp"]:
                    state["economic"]["gdp"][country] *= rng.uniform(0.85, 0.95)
            elif severity == "medium":
                # 增加特定地区政治紧张度
                regions = list(state["political"]["tensions"].keys())
                if regions:
                    target_region = rng.choice(regions)
                    state["political"]["tensions"][target_region] = min(100, state["political"]["tensions"][target_region] + rng.uniform(10, 20))
        
        elif event_type == "tech_breakthrough":
            # 技术突破影响
            if severity == "major":
                state["technological"]["ai_progress"] = min(100, state["technological"]["ai_progress"] + rng.uniform(15, 25))
                state["technological"]["renewable_energy"] = min(100, state["technological"]["renewable_energy"] + rng.uniform(10, 20))
            elif severity == "medium":
                tech_areas = ["ai_progress", "renewable_energy", "quantum_computing"]
                target_area = rng.choice(tech_areas)
                state["technological"][target_area] = min(100, state["technological"][target_area] + rng.uniform(5, 15))
        
        # 更多事件类型的处理...
        # 这里可以根据需要添加更多事件类型的处理逻辑
        
        return state
    
    def _generate_random_event(self, rng, current_time: datetime) -> Dict[str, Any]:
        """
        生成随机事件
        
        Args:
            rng: 随机数生成器
            current_time: 当前时间
            
        Returns:
            新事件
        """
        minor_events = [
            {"type": "market_fluctuation", "severity": "minor", "description": "金融市场小幅波动"},
            {"type": "policy_adjustment", "severity": "minor", "description": "某国政策小幅调整"},
            {"type": "tech_progress", "severity": "minor", "description": "技术领域小突破"},
            {"type": "diplomatic_meeting", "severity": "minor", "description": "外交会晤举行"},
            {"type": "natural_event", "severity": "minor", "description": "地区性自然灾害"}
        ]
        
        event = rng.choice(minor_events)
        event["timestamp"] = current_time.isoformat()
        event["event_id"] = str(hash(f"{current_time}{rng.random()}"))
        event["processed"] = False
        
        return event

# 测试代码
if __name__ == "__main__":
    # 简单的测试配置
    config = {
        "parallel_workers": 4,
        "max_simulation_depth": 90,
        "simulation_chunk_size": 100
    }
    
    # 创建模拟器
    simulator = WorldSimulator(config)
    
    # 创建一个简单的世界线进行测试
    from datetime import datetime
    import uuid
    
    test_worldline = {
        "id": str(uuid.uuid4()),
        "seed": 42,
        "generation": 0,
        "birth_time": datetime.now().isoformat(),
        "parent_ids": [],
        "survival_score": 0.0,
        "current_time": datetime.now().isoformat(),
        "state": {
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
        },
        "events": [
            {
                "type": "economic_crisis",
                "severity": "medium",
                "description": "测试经济危机",
                "timestamp": datetime.now().isoformat(),
                "event_id": "test_event_1"
            }
        ],
        "history": []
    }
    
    # 模拟这个世界线
    print("开始模拟测试世界线...")
    simulated = simulator.simulate([test_worldline], steps=3)
    
    # 打印结果
    print(f"模拟完成，历史记录长度: {len(simulated[0]['history'])}")
    print("最终状态:")
    print(f"经济 - GDP: {simulated[0]['state']['economic']['gdp']}")
    print(f"政治 - 紧张度: {simulated[0]['state']['political']['tensions']}")
    
    # 关闭模拟器
    simulator.shutdown()