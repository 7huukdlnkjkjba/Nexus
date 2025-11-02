#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
策略注入器模块
负责评估策略在各种世界线中的效果
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import concurrent.futures
import multiprocessing

from ..utils.logger import get_logger
from ..utils.random_utils import get_seeded_random
from ..data.data_types import WorldLine, Strategy
from ..models.interaction_model import InteractionModel

class StrategyInjector:
    """策略注入器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略注入器
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("StrategyInjector")
        self.config = config
        
        # 并行计算设置
        self.max_workers = config.get("parallel_workers", min(8, multiprocessing.cpu_count()))
        
        # 策略评估参数
        self.success_threshold = config.get("strategy_success_threshold", 0.6)  # 策略成功阈值
        
        # 初始化交互模型（用于模拟策略影响）
        self.interaction_model = InteractionModel(config)
        
        # 初始化线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="StrategyInjector"
        )
        
        self.logger.info("策略注入器初始化完成")
    
    def evaluate_strategy(self, worldlines: List[WorldLine], 
                         strategy: Dict[str, Any], 
                         simulation_steps: int = 30) -> Dict[str, Any]:
        """
        评估策略在多个世界线中的效果
        
        Args:
            worldlines: 世界线列表
            strategy: 策略定义
            simulation_steps: 注入策略后模拟的步数（天）
            
        Returns:
            策略评估结果
        """
        if not worldlines:
            self.logger.warning("没有世界线可用于策略评估")
            return {
                "success": False,
                "error": "No worldlines available",
                "total_worldlines": 0,
                "success_count": 0,
                "success_rate": 0.0,
                "key_impacts": [],
                "avg_impact_scores": {}
            }
        
        self.logger.info(f"开始评估策略 '{strategy.get('name', 'Unnamed')}'，类型: {strategy.get('type')}")
        self.logger.info(f"评估世界线数量: {len(worldlines)}, 模拟步数: {simulation_steps}")
        
        # 验证策略格式
        if not self._validate_strategy(strategy):
            self.logger.error("无效的策略格式")
            return {
                "success": False,
                "error": "Invalid strategy format",
                "total_worldlines": len(worldlines),
                "success_count": 0,
                "success_rate": 0.0,
                "key_impacts": [],
                "avg_impact_scores": {}
            }
        
        # 并行评估每个世界线
        futures = []
        chunk_size = min(100, len(worldlines) // self.max_workers + 1)
        
        for i in range(0, len(worldlines), chunk_size):
            chunk = worldlines[i:i + chunk_size]
            future = self.executor.submit(
                self._evaluate_strategy_chunk,
                chunk,
                strategy,
                simulation_steps
            )
            futures.append(future)
        
        # 收集结果
        all_results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"处理策略评估批次失败: {str(e)}")
        
        # 汇总结果
        evaluation_result = self._summarize_results(all_results, strategy)
        
        self.logger.info(f"策略评估完成，成功率: {evaluation_result['success_rate']:.4f}%")
        
        return evaluation_result
    
    def _evaluate_strategy_chunk(self, worldlines: List[WorldLine], 
                               strategy: Dict[str, Any], 
                               simulation_steps: int) -> List[Dict[str, Any]]:
        """
        评估一个批次的世界线
        
        Args:
            worldlines: 世界线批次
            strategy: 策略定义
            simulation_steps: 模拟步数
            
        Returns:
            评估结果列表
        """
        results = []
        for worldline in worldlines:
            try:
                result = self._evaluate_strategy_single_worldline(
                    worldline,
                    strategy,
                    simulation_steps
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"评估世界线 {worldline.get('id')} 失败: {str(e)}")
                # 为失败的评估提供默认结果
                results.append({
                    "worldline_id": worldline.get('id'),
                    "success": False,
                    "impact_scores": {},
                    "final_state": worldline.get('state', {})
                })
        
        return results
    
    def _evaluate_strategy_single_worldline(self, worldline: WorldLine, 
                                          strategy: Dict[str, Any], 
                                          simulation_steps: int) -> Dict[str, Any]:
        """
        在单个世界线中评估策略
        
        Args:
            worldline: 世界线
            strategy: 策略定义
            simulation_steps: 模拟步数
            
        Returns:
            评估结果
        """
        # 复制世界线以避免修改原始数据
        test_worldline = worldline.copy()
        
        # 获取随机数生成器
        rng = get_seeded_random(test_worldline["seed"])
        
        # 获取当前状态
        state = test_worldline["state"].copy()
        
        # 获取当前时间
        current_time = datetime.fromisoformat(test_worldline["current_time"])
        
        # 应用策略
        state = self._apply_strategy(state, strategy, rng)
        
        # 记录应用策略后的状态
        strategy_application_time = current_time.isoformat()
        
        # 模拟策略应用后的世界演化
        for step in range(simulation_steps):
            # 经济演化
            if "economic" in state:
                from ..models.economic_model import EconomicModel
                eco_model = EconomicModel(self.config)
                state["economic"] = eco_model.evolve(state["economic"], current_time, rng)
            
            # 政治演化
            if "political" in state:
                from ..models.political_model import PoliticalModel
                pol_model = PoliticalModel(self.config)
                state["political"] = pol_model.evolve(state["political"], current_time, rng)
            
            # 技术演化
            if "technological" in state:
                from ..models.technological_model import TechnologicalModel
                tech_model = TechnologicalModel(self.config)
                state["technological"] = tech_model.evolve(state["technological"], current_time, rng)
            
            # 气候演化
            if "climate" in state:
                from ..models.climate_model import ClimateModel
                climate_model = ClimateModel(self.config)
                state["climate"] = climate_model.evolve(state["climate"], current_time, rng)
            
            # 处理领域间的相互影响
            state = self.interaction_model.apply_interactions(state, current_time, rng)
            
            # 更新时间
            current_time += timedelta(days=1)
        
        # 评估策略效果
        impact_scores = self._calculate_impact_scores(state, strategy)
        success = self._is_strategy_successful(impact_scores, strategy)
        
        return {
            "worldline_id": test_worldline["id"],
            "success": success,
            "impact_scores": impact_scores,
            "final_state": state,
            "strategy_application_time": strategy_application_time
        }
    
    def _apply_strategy(self, state: Dict[str, Any], 
                       strategy: Dict[str, Any], 
                       rng) -> Dict[str, Any]:
        """
        应用策略到世界状态
        
        Args:
            state: 当前状态
            strategy: 策略定义
            rng: 随机数生成器
            
        Returns:
            更新后的状态
        """
        updated_state = state.copy()
        strategy_type = strategy.get("type", "")
        
        if strategy_type == "economic":
            # 经济策略
            target = strategy.get("target", "global")
            action = strategy.get("action", "")
            magnitude = strategy.get("magnitude", 1.0)
            
            if action == "tariff":
                # 关税策略
                updated_state = self._apply_tariff_strategy(updated_state, target, magnitude, rng)
            elif action == "stimulus":
                # 经济刺激策略
                updated_state = self._apply_stimulus_strategy(updated_state, target, magnitude, rng)
            elif action == "interest_rate_change":
                # 利率调整策略
                updated_state = self._apply_interest_rate_strategy(updated_state, target, magnitude, rng)
        
        elif strategy_type == "political":
            # 政治策略
            action = strategy.get("action", "")
            region = strategy.get("region", "")
            
            if action == "alliance_formation":
                # 联盟形成策略
                updated_state = self._apply_alliance_strategy(updated_state, region)
            elif action == "sanctions":
                # 制裁策略
                updated_state = self._apply_sanctions_strategy(updated_state, region, strategy.get("target_country"))
        
        elif strategy_type == "technological":
            # 技术策略
            action = strategy.get("action", "")
            field = strategy.get("field", "")
            investment = strategy.get("investment", 1.0)
            
            if action == "investment":
                # 技术投资策略
                updated_state = self._apply_tech_investment_strategy(updated_state, field, investment, rng)
        
        # 更多策略类型可以在这里添加
        
        return updated_state
    
    def _apply_tariff_strategy(self, state: Dict[str, Any], 
                              target: str, 
                              magnitude: float, 
                              rng) -> Dict[str, Any]:
        """应用关税策略"""
        if "economic" not in state:
            return state
        
        # 复制状态以避免修改原始数据
        updated = state.copy()
        
        # 关税对GDP的负面影响
        tariff_impact = -0.01 * magnitude  # 1%的关税可能导致GDP下降1%
        
        if target == "global":
            # 全球关税影响
            if "gdp" in updated["economic"]:
                for country in updated["economic"]["gdp"]:
                    # 添加一些随机性
                    variation = rng.uniform(0.8, 1.2)
                    updated["economic"]["gdp"][country] *= (1 + tariff_impact * variation)
        else:
            # 针对特定国家的关税
            if "gdp" in updated["economic"] and target in updated["economic"]["gdp"]:
                variation = rng.uniform(0.8, 1.2)
                updated["economic"]["gdp"][target] *= (1 + tariff_impact * variation)
        
        # 关税可能导致通货膨胀上升
        if "inflation" in updated["economic"]:
            for country in updated["economic"]["inflation"]:
                inflation_impact = 0.5 * magnitude  # 关税可能导致通胀上升
                updated["economic"]["inflation"][country] += inflation_impact * rng.uniform(0.8, 1.2)
        
        return updated
    
    def _apply_stimulus_strategy(self, state: Dict[str, Any], 
                                target: str, 
                                magnitude: float, 
                                rng) -> Dict[str, Any]:
        """应用经济刺激策略"""
        if "economic" not in state:
            return state
        
        updated = state.copy()
        
        # 经济刺激对GDP的正面影响
        stimulus_impact = 0.02 * magnitude  # 1单位刺激可能导致GDP上升2%
        
        if target == "global" or target in updated["economic"]["gdp"]:
            countries = list(updated["economic"]["gdp"].keys()) if target == "global" else [target]
            
            for country in countries:
                variation = rng.uniform(0.7, 1.3)
                updated["economic"]["gdp"][country] *= (1 + stimulus_impact * variation)
        
        # 经济刺激可能导致通货膨胀上升
        if "inflation" in updated["economic"]:
            countries = list(updated["economic"]["inflation"].keys()) if target == "global" else [target]
            
            for country in countries:
                inflation_impact = 0.3 * magnitude
                updated["economic"]["inflation"][country] += inflation_impact * rng.uniform(0.7, 1.3)
        
        return updated
    
    def _apply_interest_rate_strategy(self, state: Dict[str, Any], 
                                     target: str, 
                                     magnitude: float, 
                                     rng) -> Dict[str, Any]:
        """应用利率调整策略"""
        # 这里简化处理，实际应该有更复杂的模型
        return state
    
    def _apply_alliance_strategy(self, state: Dict[str, Any], 
                                region: str) -> Dict[str, Any]:
        """应用联盟形成策略"""
        if "political" not in state:
            return state
        
        updated = state.copy()
        
        # 添加新的联盟
        new_alliance = f"Alliance_{region}_{int(datetime.now().timestamp())}"
        if "alliances" not in updated["political"]:
            updated["political"]["alliances"] = []
        
        if new_alliance not in updated["political"]["alliances"]:
            updated["political"]["alliances"].append(new_alliance)
        
        # 减少相关地区的紧张度
        if "tensions" in updated["political"] and region in updated["political"]["tensions"]:
            updated["political"]["tensions"][region] = max(0, updated["political"]["tensions"][region] - 20)
        
        return updated
    
    def _apply_sanctions_strategy(self, state: Dict[str, Any], 
                                region: str, 
                                target_country: str) -> Dict[str, Any]:
        """应用制裁策略"""
        if "political" not in state or "economic" not in state:
            return state
        
        updated = state.copy()
        
        # 增加政治紧张度
        if "tensions" in updated["political"]:
            if region not in updated["political"]["tensions"]:
                updated["political"]["tensions"][region] = 50
            updated["political"]["tensions"][region] = min(100, updated["political"]["tensions"][region] + 30)
        
        # 影响目标国家的经济
        if "gdp" in updated["economic"] and target_country in updated["economic"]["gdp"]:
            updated["economic"]["gdp"][target_country] *= 0.9  # 制裁导致GDP下降10%
        
        return updated
    
    def _apply_tech_investment_strategy(self, state: Dict[str, Any], 
                                      field: str, 
                                      investment: float, 
                                      rng) -> Dict[str, Any]:
        """应用技术投资策略"""
        if "technological" not in state:
            return state
        
        updated = state.copy()
        
        # 增加技术领域的进展
        if field in updated["technological"]:
            current_level = updated["technological"][field]
            # 投资带来的技术进步（上限为100）
            progress = investment * rng.uniform(5, 15)
            updated["technological"][field] = min(100, current_level + progress)
        
        return updated
    
    def _calculate_impact_scores(self, state: Dict[str, Any], 
                               strategy: Dict[str, Any]) -> Dict[str, float]:
        """
        计算策略对各个领域的影响分数
        
        Args:
            state: 最终状态
            strategy: 策略定义
            
        Returns:
            影响分数字典
        """
        impact_scores = {
            "economic": 0.0,
            "political": 0.0,
            "technological": 0.0,
            "climate": 0.0,
            "overall": 0.0
        }
        
        # 根据策略目标计算影响分数
        strategy_type = strategy.get("type", "")
        objectives = strategy.get("objectives", [])
        
        # 这里是简化的影响计算，实际应该基于策略目标和状态变化
        # 例如，如果策略目标是"increase_gdp"，则检查GDP是否增加
        
        # 示例逻辑：
        if strategy_type == "economic":
            impact_scores["economic"] = 0.7  # 假设经济策略主要影响经济
            impact_scores["overall"] = 0.65
        elif strategy_type == "political":
            impact_scores["political"] = 0.8  # 假设政治策略主要影响政治
            impact_scores["overall"] = 0.7
        elif strategy_type == "technological":
            impact_scores["technological"] = 0.9  # 假设技术策略主要影响技术
            impact_scores["overall"] = 0.8
        
        return impact_scores
    
    def _is_strategy_successful(self, impact_scores: Dict[str, float], 
                              strategy: Dict[str, Any]) -> bool:
        """
        判断策略是否成功
        
        Args:
            impact_scores: 影响分数
            strategy: 策略定义
            
        Returns:
            是否成功
        """
        # 基于总体影响分数判断
        overall_score = impact_scores.get("overall", 0.0)
        
        # 如果策略有特定的成功条件，可以在这里添加
        specific_conditions = strategy.get("success_conditions", [])
        
        # 简单判断：如果总体分数高于阈值，则认为成功
        return overall_score >= self.success_threshold
    
    def _validate_strategy(self, strategy: Dict[str, Any]) -> bool:
        """
        验证策略格式是否有效
        
        Args:
            strategy: 策略定义
            
        Returns:
            是否有效
        """
        # 基本验证
        required_fields = ["type", "action"]
        
        for field in required_fields:
            if field not in strategy:
                self.logger.error(f"策略缺少必要字段: {field}")
                return False
        
        # 验证策略类型
        valid_types = ["economic", "political", "technological", "climate", "hybrid"]
        strategy_type = strategy.get("type")
        
        if strategy_type not in valid_types:
            self.logger.error(f"无效的策略类型: {strategy_type}")
            return False
        
        return True
    
    def _summarize_results(self, results: List[Dict[str, Any]], 
                          strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        汇总所有世界线的评估结果
        
        Args:
            results: 评估结果列表
            strategy: 策略定义
            
        Returns:
            汇总结果
        """
        total = len(results)
        success_count = sum(1 for r in results if r.get("success", False))
        success_rate = (success_count / total * 100) if total > 0 else 0.0
        
        # 计算平均影响分数
        avg_impact_scores = {
            "economic": 0.0,
            "political": 0.0,
            "technological": 0.0,
            "climate": 0.0,
            "overall": 0.0
        }
        
        if results:
            for category in avg_impact_scores:
                scores = [r["impact_scores"].get(category, 0.0) for r in results]
                avg_impact_scores[category] = np.mean(scores)
        
        # 识别关键影响
        key_impacts = []
        
        # 基于策略类型和平均影响分数确定关键影响
        if strategy.get("type") == "economic":
            if avg_impact_scores["economic"] > 0.5:
                key_impacts.append({
                    "area": "economic",
                    "description": "经济策略产生了显著影响",
                    "magnitude": "high"
                })
        elif strategy.get("type") == "political":
            if avg_impact_scores["political"] > 0.5:
                key_impacts.append({
                    "area": "political",
                    "description": "政治策略改变了地缘政治格局",
                    "magnitude": "high"
                })
        
        return {
            "success": True,
            "total_worldlines": total,
            "success_count": success_count,
            "success_rate": success_rate,
            "key_impacts": key_impacts,
            "avg_impact_scores": avg_impact_scores,
            "strategy": strategy
        }

# 测试代码
if __name__ == "__main__":
    # 简单的测试配置
    config = {
        "parallel_workers": 4,
        "strategy_success_threshold": 0.6
    }
    
    # 创建策略注入器
    injector = StrategyInjector(config)
    
    # 创建测试世界线
    from datetime import datetime
    import uuid
    
    worldlines = []
    
    # 创建10个世界线
    for i in range(10):
        worldline = {
            "id": str(uuid.uuid4()),
            "seed": 42 + i,
            "current_time": datetime.now().isoformat(),
            "state": {
                "economic": {
                    "gdp": {"US": 25.0 + i, "China": 18.0 - i, "Japan": 4.0 + i * 0.1},
                    "inflation": {"US": 3.0 + i * 0.1, "EU": 4.0 - i * 0.1},
                    "unemployment": {"US": 4.0 - i * 0.1, "EU": 7.0 + i * 0.2}
                },
                "political": {
                    "alliances": ["NATO", "EU"],
                    "tensions": {"RegionA": 50 + i * 5, "RegionB": 60 - i * 3}
                },
                "technological": {
                    "ai_progress": 70 + i * 2,
                    "renewable_energy": 45 - i,
                    "quantum_computing": 30 + i * 1.5
                }
            }
        }
        worldlines.append(worldline)
    
    # 定义测试策略
    tariff_strategy = {
        "name": "对中国加征关税",
        "type": "economic",
        "action": "tariff",
        "target": "China",
        "magnitude": 2.0,  # 200%的关税增幅
        "objectives": ["reduce_trade_deficit", "protect_domestic_industry"],
        "success_conditions": ["gdp_impact_positive"]
    }
    
    # 评估策略
    print("开始评估关税策略...")
    result = injector.evaluate_strategy(worldlines, tariff_strategy, simulation_steps=10)
    
    # 打印结果
    print(f"\n评估结果:")
    print(f"总世界线数: {result['total_worldlines']}")
    print(f"成功世界线数: {result['success_count']}")
    print(f"成功率: {result['success_rate']:.2f}%")
    print(f"\n平均影响分数:")
    print(f"经济: {result['avg_impact_scores']['economic']:.4f}")
    print(f"政治: {result['avg_impact_scores']['political']:.4f}")
    print(f"技术: {result['avg_impact_scores']['technological']:.4f}")
    print(f"总体: {result['avg_impact_scores']['overall']:.4f}")
    print(f"\n关键影响:")
    for impact in result['key_impacts']:
        print(f"- {impact['description']} (领域: {impact['area']}, 程度: {impact['magnitude']})")