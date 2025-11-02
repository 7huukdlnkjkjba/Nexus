"""
模拟协调器模块

负责协调各个子模型之间的交互，管理整个模拟流程，
是系统的核心调度中心。
"""

import time
import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np

from ..data.data_types import (
    WorldLine, DataPoint, RealityData
)
from ..utils.data_types import WorldEvent as Event, EventType
from ..models.evaluator import WorldLineEvaluator
from ..models.economic_model import EconomicModel
from ..models.political_model import PoliticalModel
from ..models.technology_model import TechnologyModel
from ..models.climate_model import ClimateModel
from ..models.crisis_response_model import CrisisResponseModel
from .event_system import EventSystem
from .timeline_manager import TimelineManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimulationCoordinator:
    """
    模拟协调器类
    
    负责初始化和协调各个子模型，管理模拟流程，处理模型间的信息传递。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模拟协调器
        
        Args:
            config: 配置字典，包含所有子模型的配置
        """
        # 设置随机种子，确保可复现性
        self.seed = config.get("seed", int(time.time()))
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # 模拟参数
        self.initial_year = config.get("initial_year", 2023)
        self.target_year = config.get("target_year", 2100)
        self.max_timelines = config.get("max_timelines", 20)
        self.timeline_pruning_threshold = config.get("timeline_pruning_threshold", 0.1)
        self.decision_points_per_year = config.get("decision_points_per_year", 4)
        
        # 用于存储运行时间统计
        self.execution_times = {}
        
        # 初始化各个子模型
        self._initialize_models(config)
        
        # 初始化世界线管理器
        self.timeline_manager = TimelineManager(config.get("timeline_manager", {}))
        
        # 初始化世界线评估器
        self.evaluator = WorldLineEvaluator(config.get("evaluator", {}))
        
        # 初始化第一个世界线作为基准线
        self._initialize_base_timeline()
        
        # 模拟状态
        self.is_running = False
        self.current_time = self.initial_year
        self.simulation_step = 0
        
        logger.info(f"模拟协调器初始化完成。初始年份: {self.initial_year}, 目标年份: {self.target_year}, 随机种子: {self.seed}")
    
    def _initialize_models(self, config: Dict[str, Any]) -> None:
        """
        初始化各个子模型
        
        Args:
            config: 配置字典
        """
        start_time = time.time()
        
        # 初始化经济模型
        self.economic_model = EconomicModel(config.get("economic_model", {}))
        
        # 初始化政治模型
        self.political_model = PoliticalModel(config.get("political_model", {}))
        
        # 初始化技术模型
        self.technology_model = TechnologyModel(config.get("technology_model", {}))
        
        # 初始化气候模型
        self.climate_model = ClimateModel(config.get("climate_model", {}))
        
        # 初始化事件系统
        self.event_system = EventSystem(config.get("event_system", {}))
        
        # 初始化危机响应模型
        self.crisis_response_model = CrisisResponseModel(config.get("crisis_response_model", {}))
        
        self.execution_times["model_initialization"] = time.time() - start_time
        logger.info(f"所有子模型初始化完成，耗时: {self.execution_times['model_initialization']:.2f} 秒")
    
    def _initialize_base_timeline(self) -> None:
        """
        初始化基础世界线
        """
        # 初始化各种状态数据
        economic_state = self.economic_model.initialize_state(self.initial_year)
        political_state = self.political_model.initialize_state(self.initial_year)
        technology_state = self.technology_model.initialize_state(self.initial_year)
        climate_state = self.climate_model.initialize_state(self.initial_year)
        
        # 创建状态字典，包含所有领域的状态
        state = {
            "economic": economic_state,
            "political": political_state,
            "technology": technology_state,
            "climate": climate_state
        }
        
        # 创建基础世界线
        base_timeline = WorldLine(
            id="base_timeline",
            current_time=str(self.initial_year),
            seed=self.seed,
            generation=0,
            birth_time=str(self.initial_year),
            state=state,
            events=[],
            history=[]
        )
        
        # 将基础世界线添加到管理器
        self.timeline_manager.add_worldline(base_timeline)
        
        logger.info("基础世界线初始化完成")
    
    def _process_single_timeline(self, timeline: WorldLine, year: int, decision_point: int) -> WorldLine:
        """
        处理单个世界线（用于并行处理）
        
        Args:
            timeline: 要处理的世界线
            year: 当前年份
            decision_point: 当前决策点索引
            
        Returns:
            处理后的世界线
        """
        # 创建时间线的深拷贝以避免多进程中的共享内存问题
        timeline_copy = timeline.copy()
        
        # 使用基于时间线ID和年份的种子确保确定性
        seed = hash(f"{timeline.id}_{year}_{decision_point}")
        
        # 更新世界线状态（不在这里进行评估，留待批量处理）
        updated_timeline = self._update_timeline(timeline_copy, year, decision_point, seed)
        
        return updated_timeline
    
    def run_simulation(self) -> List[WorldLine]:
        """
        运行完整的模拟（使用并行处理优化性能）
        
        Returns:
            模拟结束后的世界线列表
        """
        self.is_running = True
        start_time = time.time()
        
        try:
            logger.info(f"开始模拟: {self.initial_year} -> {self.target_year}")
            
            # 逐年运行模拟
            for year in range(self.initial_year, self.target_year):
                self.current_time = year
                
                # 在每一年中处理多个决策点
                for decision_point in range(self.decision_points_per_year):
                    self.simulation_step += 1
                    logger.info(f"处理决策点: 年份 {year}, 第 {decision_point+1}/{self.decision_points_per_year} 季度")
                    
                    # 获取当前所有世界线
                    timelines = self.timeline_manager.get_all_worldlines()
                    
                    # 并行更新世界线
                    updated_timelines = []
                    process_count = min(16, len(timelines))  # 根据系统资源调整
                    
                    if process_count > 1:
                        # 使用并行处理
                        process_timeline_func = partial(
                            self._process_single_timeline,
                            year=year,
                            decision_point=decision_point
                        )
                        
                        with ProcessPoolExecutor(max_workers=process_count) as executor:
                            future_to_timeline = {executor.submit(process_timeline_func, timeline): timeline for timeline in timelines}
                            for future in as_completed(future_to_timeline):
                                try:
                                    updated_timeline = future.result()
                                    updated_timelines.append(updated_timeline)
                                except Exception as e:
                                    logger.error(f"处理世界线时出错: {str(e)}")
                    else:
                        # 单线程处理（世界线数量较少时）
                        for timeline in timelines:
                            updated_timeline = self._process_single_timeline(timeline, year, decision_point)
                            updated_timelines.append(updated_timeline)
                    
                    # 使用批量评估优化性能
                    if hasattr(self.evaluator, 'batch_evaluate'):
                        try:
                            evaluation_results = self.evaluator.batch_evaluate(updated_timelines)
                            for timeline in updated_timelines:
                                timeline.survival_probability = evaluation_results[timeline.id]["survival_probability"]
                                timeline.value_score = evaluation_results[timeline.id]["value_score"]
                        except Exception as e:
                            logger.error(f"批量评估世界线时出错: {str(e)}")
                            # 回退到单个评估
                            for timeline in updated_timelines:
                                evaluation = self.evaluator.evaluate_worldline(timeline)
                                timeline.survival_probability = evaluation["survival_probability"]
                                timeline.value_score = evaluation["value_score"]
                    else:
                        # 单个评估
                        for timeline in updated_timelines:
                            evaluation = self.evaluator.evaluate_worldline(timeline)
                            timeline.survival_probability = evaluation["survival_probability"]
                            timeline.value_score = evaluation["value_score"]
                    
                    # 更新时间线管理器中的世界线
                    for timeline in updated_timelines:
                        self.timeline_manager.add_worldline(timeline)
                    
                    # 检查是否需要分叉世界线
                    if decision_point == self.decision_points_per_year - 1:  # 每年只在最后一个决策点分叉
                        self._branch_timelines(year)
                    
                    # 修剪低概率世界线
                    self._prune_timelines()
                    
                    # 检查是否达到最大世界线数量
                    self._check_timeline_limit()
            
            self.execution_times["total_simulation"] = time.time() - start_time
            logger.info(f"模拟完成！总耗时: {self.execution_times['total_simulation']:.2f} 秒")
            
        except Exception as e:
            logger.error(f"模拟过程中出错: {str(e)}")
            raise
        finally:
            self.is_running = False
        
        return self.timeline_manager.get_all_worldlines()
    
    def _update_timeline(self, timeline: WorldLine, year: int, decision_point: int, seed: Optional[int] = None) -> WorldLine:
        """
        更新单个世界线的状态（优化版本，支持自定义种子）
        
        Args:
            timeline: 要更新的世界线
            year: 当前年份
            decision_point: 当前决策点索引
            seed: 随机种子（可选）
            
        Returns:
            更新后的世界线
        """
        # 更新当前时间（直接修改）
        timeline.current_time = year + (decision_point + 1) / self.decision_points_per_year
        
        # 生成并处理事件 - 使用优化的事件系统
        events = self._generate_and_process_events(timeline, year, decision_point, seed)
        
        # 处理事件响应
        responses = self._process_event_responses(timeline, events)
        
        # 批量更新各领域状态以提高性能
        domain_updates = {
            "economic": self._update_economic_state(timeline.state.get("economic", {}), events, responses, year, decision_point),
            "political": self._update_political_state(timeline.state.get("political", {}), events, responses, year, decision_point),
            "technology": self._update_technology_state(timeline.state.get("technology", {}), events, responses, year, decision_point),
            "climate": self._update_climate_state(timeline.state.get("climate", {}), events, responses, year, decision_point)
        }
        
        # 一次性更新所有领域状态
        timeline.economic_state = domain_updates["economic"]
        timeline.political_state = domain_updates["political"]
        timeline.technology_state = domain_updates["technology"]
        timeline.climate_state = domain_updates["climate"]
        
        # 记录历史
        if decision_point == self.decision_points_per_year - 1 or len(timeline.history) >= 10:  # 年末或历史记录过多时
            # 创建历史记录条目
            history_entry = {
                "year": year,
                "decision_point": decision_point,
                "state": timeline.state.copy(),
                "events": events.copy()
            }
            
            # 添加到历史记录列表
            timeline.history.append(history_entry)
            
            # 清理旧的历史记录（保留最近的20条记录）
            if len(timeline.history) > 20:
                timeline.history = timeline.history[-20:]
        
        # 优化事件的存储：只保留关键事件
        important_events = [event for event in events if event.severity in [EventSeverity.CRITICAL, EventSeverity.HIGH]]
        timeline.events.extend(important_events)
        
        # 限制事件历史记录大小
        if len(timeline.events) > 500:
            timeline.events = timeline.events[-500:]
        
        return timeline
    
    def _create_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建状态的轻量级摘要，减少内存使用
        
        Args:
            state: 完整状态字典
            
        Returns:
            精简的状态摘要
        """
        # 根据不同类型的状态提取关键指标
        # 这里是一个通用实现，各个模型可以根据需要自定义
        summary = {}
        
        # 提取数值型指标（通常是最关键的）
        for key, value in state.items():
            if isinstance(value, (int, float, bool)):
                summary[key] = value
            elif isinstance(value, str) and len(value) < 100:  # 保留简短的字符串信息
                summary[key] = value
            # 对于复杂结构，只提取第一层的关键信息
            elif isinstance(value, dict):
                # 尝试提取子字典中的数值型指标
                sub_summary = {k: v for k, v in value.items() if isinstance(v, (int, float, bool))}
                if sub_summary:
                    summary[key] = sub_summary
        
        return summary
    
    def _generate_and_process_events(self, timeline: WorldLine, year: int, decision_point: int, seed: Optional[int] = None) -> List[Event]:
        """
        生成并处理事件（支持自定义种子）
        
        Args:
            timeline: 当前世界线
            year: 当前年份
            decision_point: 当前决策点索引
            seed: 随机种子（可选）
            
        Returns:
            生成的事件列表
        """
        # 创建一个具有year和month属性的时间对象
        class TimeObject:
            def __init__(self, year, month):
                self.year = year
                self.month = month
        
        # 构建事件生成上下文
        context = {
            "current_time": TimeObject(year, (decision_point + 1) * 3),  # 假设每个决策点代表一个季度
            "economic_state": timeline.state.get("economic", {}),
            "political_state": timeline.state.get("political", {}),
            "technology_state": timeline.state.get("technology", {}),
            "climate_state": timeline.state.get("climate", {}),
            "events_history": timeline.events,
            "decision_point": decision_point,
            "decision_points_per_year": self.decision_points_per_year
        }
        
        # 使用提供的种子或默认种子
        event_seed = seed if seed is not None else (timeline.seed + self.simulation_step)
        
        # 从context中提取各个状态
        current_time = context["current_time"]
        economic_state = context["economic_state"]
        political_state = context["political_state"]
        technology_state = context["technology_state"]
        climate_state = context["climate_state"]
        
        # 直接传递各个参数给事件系统
        events = self.event_system.generate_events(
            current_time,
            economic_state,
            political_state,
            technology_state,
            climate_state,
            event_seed
        )
        
        # 处理事件对世界线的影响
        for event in events:
            # 这里可以添加额外的事件处理逻辑
            logger.info(f"生成事件: {event.type} - {event.description}, 严重程度: {event.severity}")
        
        return events
    
    def _process_event_responses(self, timeline: WorldLine, events: List[Event]) -> List[Dict[str, Any]]:
        """
        处理事件响应
        
        Args:
            timeline: 当前世界线
            events: 事件列表
            
        Returns:
            响应列表
        """
        all_responses = []
        
        for event in events:
            # 生成响应
            responses = self.crisis_response_model.generate_responses(
                event,
                timeline.economic_state,
                timeline.political_state,
                timeline.climate_state,
                seed=timeline.seed + len(timeline.responses)
            )
            
            all_responses.extend(responses)
            
            # 记录响应历史
            for response in responses:
                self.crisis_response_model.record_response(event, response)
        
        return all_responses
    
    def _update_economic_state(self, economic_state: Dict[str, Any], 
                              events: List[Event], 
                              responses: List[Dict[str, Any]],
                              year: int,
                              decision_point: int) -> Dict[str, Any]:
        """
        更新经济状态
        
        Args:
            economic_state: 当前经济状态
            events: 事件列表
            responses: 响应列表
            year: 当前年份
            decision_point: 当前决策点索引
            
        Returns:
            更新后的经济状态
        """
        # 构建时间对象
        from datetime import datetime
        current_time = datetime(year, (decision_point % 4) * 3 + 1, 1)
        
        # 获取随机数生成器
        from ..utils.random_utils import get_seeded_random
        rng = get_seeded_random(seed=hash(f"{year}-{decision_point}-economic"))
        
        # 更新经济状态，使用evolve方法
        updated_state = self.economic_model.evolve(
            economic_state=economic_state,
            current_time=current_time,
            rng=rng
        )
        
        return updated_state
    
    def _update_political_state(self, political_state: Dict[str, Any],
                               events: List[Event],
                               responses: List[Dict[str, Any]],
                               year: int,
                               decision_point: int) -> Dict[str, Any]:
        """
        更新政治状态
        
        Args:
            political_state: 当前政治状态
            events: 事件列表
            responses: 响应列表
            year: 当前年份
            decision_point: 当前决策点索引
            
        Returns:
            更新后的政治状态
        """
        # 构建时间对象
        from datetime import datetime
        current_time = datetime(year, (decision_point % 4) * 3 + 1, 1)
        
        # 获取随机数生成器
        from ..utils.random_utils import get_seeded_random
        rng = get_seeded_random(seed=hash(f"{year}-{decision_point}-political"))
        
        # 获取经济状态用于演化（默认为空字典）
        economic_state = {}
        
        # 更新政治状态，使用evolve方法而不是evolve_state
        updated_state = self.political_model.evolve(
            political_state=political_state,
            current_time=current_time,
            economic_state=economic_state,
            rng=rng
        )
        
        # 考虑响应后的国际关系变化
        for response in responses:
            if "actor_type" in response and response["actor_type"] == "country":
                # 获取国家列表
                countries = list(self.political_model.country_stability.keys())
                
                # 更新关系矩阵
                relation_changes = self.crisis_response_model.update_relations_after_response(
                    response, 
                    countries
                )
                
                # 应用关系变化
                for (country1, country2), change in relation_changes.items():
                    if country1 in updated_state["relations"] and country2 in updated_state["relations"][country1]:
                        updated_state["relations"][country1][country2] += change
                        # 确保关系值在合理范围内
                        updated_state["relations"][country1][country2] = min(
                            max(updated_state["relations"][country1][country2], -1.0), 
                            1.0
                        )
        
        return updated_state
    
    def _update_technology_state(self, technology_state: Dict[str, Any],
                                events: List[Event],
                                responses: List[Dict[str, Any]],
                                year: int,
                                decision_point: int) -> Dict[str, Any]:
        """
        更新技术状态
        
        Args:
            technology_state: 当前技术状态
            events: 事件列表
            responses: 响应列表
            year: 当前年份
            decision_point: 当前决策点索引
            
        Returns:
            更新后的技术状态
        """
        # 构建时间对象
        from datetime import datetime
        current_time = datetime(year, (decision_point % 4) * 3 + 1, 1)
        
        # 获取随机数生成器
        from ..utils.random_utils import get_seeded_random
        rng = get_seeded_random(seed=hash(f"{year}-{decision_point}-technology"))
        
        # 获取经济状态用于演化（默认为空字典）
        economic_state = {}
        
        # 使用正确的方法名
        updated_state = self.technology_model.evolve(
            technology_state=technology_state,
            current_time=current_time,
            economic_state=economic_state,
            rng=rng
        )
        
        return updated_state
    
    def _update_climate_state(self, climate_state: Dict[str, Any],
                             events: List[Event],
                             responses: List[Dict[str, Any]],
                             year: int,
                             decision_point: int) -> Dict[str, Any]:
        """
        更新气候状态
        
        Args:
            climate_state: 当前气候状态
            events: 事件列表
            responses: 响应列表
            year: 当前年份
            decision_point: 当前决策点索引
            
        Returns:
            更新后的气候状态
        """
        # 构建时间对象
        from datetime import datetime
        current_time = datetime(year, (decision_point % 4) * 3 + 1, 1)
        
        # 获取随机数生成器
        from ..utils.random_utils import get_seeded_random
        rng = get_seeded_random(seed=hash(f"{year}-{decision_point}-climate"))
        
        # 获取其他状态用于演化（默认为空字典）
        economic_state = {}
        technology_state = {}
        political_state = {}
        
        # 使用正确的方法名
        updated_state = self.climate_model.evolve(
            climate_state=climate_state,
            current_time=current_time,
            economic_state=economic_state,
            technology_state=technology_state,
            political_state=political_state,
            rng=rng
        )
        
        return updated_state
    
    def _branch_timelines(self, year: int) -> None:
        """
        分叉世界线
        
        Args:
            year: 当前年份
        """
        # 获取当前所有世界线
        timelines = self.timeline_manager.get_all_worldlines()
        
        # 按价值分数排序
        timelines.sort(key=lambda t: t.value_score, reverse=True)
        
        # 只为前N个高分世界线创建分叉
        num_to_branch = min(3, len(timelines))
        
        for i in range(num_to_branch):
            timeline = timelines[i]
            
            # 决定是否分叉
            if random.random() < 0.7:  # 70%的概率分叉
                # 创建分叉
                branch1 = timeline.copy()
                branch2 = timeline.copy()
                
                # 为分叉设置新ID和种子
                branch1.id = f"{timeline.id}_branch1_y{year}"
                branch1.seed = random.randint(1, 1000000)
                branch1.generation = timeline.generation + 1
                branch1.birth_time = year
                
                branch2.id = f"{timeline.id}_branch2_y{year}"
                branch2.seed = random.randint(1, 1000000)
                branch2.generation = timeline.generation + 1
                branch2.birth_time = year
                
                # 在分叉世界线中引入一些微小的差异
                self._introduce_branch_differences(branch1, branch2)
                
                # 添加分叉世界线
                self.timeline_manager.add_worldline(branch1)
                self.timeline_manager.add_worldline(branch2)
                
                logger.info(f"世界线分叉: {timeline.id} 在年份 {year} 创建了两个分支")
    
    def _introduce_branch_differences(self, branch1: WorldLine, branch2: WorldLine) -> None:
        """
        在分叉世界线中引入差异
        
        Args:
            branch1: 第一个分支
            branch2: 第二个分支
        """
        # 经济状态差异
        diff_factor = random.uniform(0.01, 0.05)
        
        # 调整GDP增长率
        if "global_gdp_growth" in branch1.economic_state:
            branch1.economic_state["global_gdp_growth"] *= (1 + diff_factor)
            branch2.economic_state["global_gdp_growth"] *= (1 - diff_factor)
        
        # 调整通胀率
        if "inflation_rate" in branch1.economic_state:
            branch1.economic_state["inflation_rate"] *= (1 + diff_factor)
            branch2.economic_state["inflation_rate"] *= (1 - diff_factor)
        
        # 政治状态差异
        if "international_cooperation" in branch1.political_state:
            branch1.political_state["international_cooperation"] *= (1 + diff_factor)
            branch2.political_state["international_cooperation"] *= (1 - diff_factor)
        
        # 技术研发投入差异
        if "global_research_funding" in branch1.technology_state:
            branch1.technology_state["global_research_funding"] *= (1 + diff_factor)
            branch2.technology_state["global_research_funding"] *= (1 - diff_factor)
        
        # 确保数值在合理范围内
        for state in [branch1.economic_state, branch2.economic_state, 
                     branch1.political_state, branch2.political_state,
                     branch1.technology_state, branch2.technology_state]:
            for key, value in state.items():
                if isinstance(value, float):
                    if "growth" in key or "rate" in key:
                        state[key] = max(min(value, 10.0), -10.0)
                    elif "cooperation" in key or "funding" in key:
                        state[key] = max(min(value, 1.0), 0.0)
    
    def _prune_timelines(self) -> None:
        """
        修剪低概率世界线
        """
        # 获取所有世界线
        timelines = self.timeline_manager.get_all_worldlines()
        
        # 过滤掉生存概率低于阈值的世界线
        to_remove = [t for t in timelines if t.survival_probability < self.timeline_pruning_threshold]
        
        for timeline in to_remove:
            self.timeline_manager.remove_timeline(timeline.id)
            logger.info(f"世界线被修剪: {timeline.id}, 生存概率: {timeline.survival_probability:.4f}")
    
    def _check_timeline_limit(self) -> None:
        """
        检查是否达到最大世界线数量限制
        """
        # 获取所有世界线并排序
        timelines = self.timeline_manager.get_all_worldlines()
        
        if len(timelines) > self.max_timelines:
            # 按生存概率和价值分数的综合分数排序
            timelines.sort(key=lambda t: t.survival_probability * 0.7 + t.value_score * 0.3, reverse=True)
            
            # 删除分数最低的多余世界线
            to_remove = timelines[self.max_timelines:]
            
            for timeline in to_remove:
                self.timeline_manager.remove_timeline(timeline.id)
                logger.info(f"世界线被限制删除: {timeline.id}, 综合分数: {timeline.survival_probability * 0.7 + timeline.value_score * 0.3:.4f}")
    
    def get_simulation_results(self) -> Dict[str, Any]:
        """
        获取模拟结果
        
        Returns:
            模拟结果字典
        """
        timelines = self.timeline_manager.get_all_worldlines()
        
        # 为每个世界线生成评估
        timeline_evaluations = {}
        for timeline in timelines:
            evaluation = self.evaluator.evaluate_worldline(timeline)
            timeline_evaluations[timeline.id] = evaluation
        
        # 初始化事件统计信息（简化版）
        event_statistics = {"total_events": 0, "event_types": {}}
        
        # 获取响应统计
        response_statistics = self.crisis_response_model.get_response_statistics()
        
        return {
            "timelines": timelines,
            "timeline_evaluations": timeline_evaluations,
            "event_statistics": event_statistics,
            "response_statistics": response_statistics,
            "execution_times": self.execution_times,
            "simulation_parameters": {
                "initial_year": self.initial_year,
                "target_year": self.target_year,
                "max_timelines": self.max_timelines,
                "seed": self.seed,
                "decision_points_per_year": self.decision_points_per_year
            }
        }
    
    def stop_simulation(self) -> None:
        """
        停止模拟
        """
        self.is_running = False
        logger.info("模拟已停止")
    
    def save_simulation_state(self, filepath: str) -> bool:
        """
        保存模拟状态到文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否保存成功
        """
        import pickle
        
        try:
            # 构建要保存的状态
            state = {
                "current_time": self.current_time,
                "simulation_step": self.simulation_step,
                "timelines": self.timeline_manager.get_all_timelines(),
                "seed": self.seed,
                "execution_times": self.execution_times
            }
            
            # 保存到文件
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"模拟状态已保存到 {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存模拟状态时出错: {str(e)}")
            return False
    
    def load_simulation_state(self, filepath: str) -> bool:
        """
        从文件加载模拟状态
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否加载成功
        """
        import pickle
        
        try:
            # 从文件加载
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # 恢复状态
            self.current_time = state["current_time"]
            self.simulation_step = state["simulation_step"]
            self.seed = state["seed"]
            self.execution_times = state["execution_times"]
            
            # 恢复世界线
            for timeline in state["timelines"]:
                self.timeline_manager.add_timeline(timeline)
            
            logger.info(f"模拟状态已从 {filepath} 加载")
            return True
        except Exception as e:
            logger.error(f"加载模拟状态时出错: {str(e)}")
            return False


# 测试代码
if __name__ == "__main__":
    # 创建模拟配置
    config = {
        "seed": 42,
        "initial_year": 2023,
        "target_year": 2025,  # 测试用短时间范围
        "max_timelines": 5,
        "timeline_pruning_threshold": 0.1,
        "decision_points_per_year": 2,  # 每年两个决策点以加快测试
        
        # 子模型配置
        "economic_model": {
            "initial_gdp": 100.0,
            "gdp_growth_volatility": 0.02,
            "inflation_base_rate": 0.02,
            "technology_contribution": 0.3
        },
        "political_model": {
            "base_stability": 0.7,
            "conflict_impact": 0.2
        },
        "technology_model": {
            "base_research_intensity": 0.025,
            "breakthrough_probability": 0.05
        },
        "climate_model": {
            "base_emission_rate": 50.0,
            "temperature_sensitivity": 0.7
        },
        "event_system": {
            "base_event_probability": 0.1,
            "max_events_per_period": 2
        },
        "crisis_response_model": {
            "max_country_responses_per_event": 3,
            "max_organization_responses_per_event": 2
        },
        "timeline_manager": {
            "cache_size": 100
        },
        "evaluator": {
            "survival_weight": 0.6,
            "value_weight": 0.4
        }
    }
    
    # 创建模拟协调器
    coordinator = SimulationCoordinator(config)
    
    print("\n===== 开始运行模拟测试 =====")
    
    # 运行模拟
    try:
        results = coordinator.run_simulation()
        
        print(f"\n模拟完成！生成了 {len(results)} 个世界线")
        
        # 获取详细结果
        detailed_results = coordinator.get_simulation_results()
        
        print("\n===== 模拟结果统计 =====")
        
        # 打印世界线评估
        print("\n世界线评估:")
        for timeline_id, eval in detailed_results["timeline_evaluations"].items():
            print(f"{timeline_id}:")
            print(f"  生存概率: {eval['survival_probability']:.4f}")
            print(f"  价值分数: {eval['value_score']:.4f}")
            print(f"  经济评分: {eval['domain_scores']['economic']:.4f}")
            print(f"  政治评分: {eval['domain_scores']['political']:.4f}")
            print(f"  技术评分: {eval['domain_scores']['technology']:.4f}")
            print(f"  气候评分: {eval['domain_scores']['climate']:.4f}")
            print()
        
        # 打印事件统计
        print("\n事件统计:")
        for event_type, count in detailed_results["event_statistics"].get("events_by_type", {}).items():
            print(f"  {event_type}: {count}")
        
        # 打印执行时间
        print("\n执行时间统计:")
        for phase, duration in detailed_results["execution_times"].items():
            print(f"  {phase}: {duration:.2f} 秒")
            
    except Exception as e:
        print(f"模拟测试失败: {str(e)}")
    
    print("\n模拟协调器测试完成！")