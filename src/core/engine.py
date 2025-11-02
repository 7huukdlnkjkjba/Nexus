#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
核心引擎类
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..data.data_collector import DataCollector
from ..data.data_processor import DataProcessor
from .seed_generator import SeedGenerator
from .world_simulator import WorldSimulator
from .reality_comparator import RealityComparator
from .survivor_selector import SurvivorSelector
from .strategy_injector import StrategyInjector
from .insight_extractor import InsightExtractor
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector

class NexusEngine:
    """Nexus引擎主类，协调所有核心组件的工作"""
    
    def __init__(self, config: Dict[str, Any], mode: str = "standard"):
        """
        初始化Nexus引擎
        
        Args:
            config: 配置字典
            mode: 运行模式: standard, lightweight, heavy
        """
        self.logger = get_logger("NexusEngine")
        self.config = config
        self.mode = mode
        self.running = False
        self.engine_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 根据模式调整配置
        self._adjust_config_for_mode()
        
        # 初始化核心组件
        self.logger.info("初始化核心组件...")
        self.data_collector = DataCollector(config)
        self.data_processor = DataProcessor(config)
        self.seed_generator = SeedGenerator(config)
        self.world_simulator = WorldSimulator(config)
        self.reality_comparator = RealityComparator(config)
        self.survivor_selector = SurvivorSelector(config)
        self.strategy_injector = StrategyInjector(config)
        self.insight_extractor = InsightExtractor(config)
        self.metrics = MetricsCollector()
        
        # 世界线存储
        self.active_worldlines = []
        self.survivor_worldlines = []
        self.current_reality_data = None
        
        self.logger.info("引擎初始化完成")
    
    def _adjust_config_for_mode(self):
        """根据运行模式调整配置"""
        if self.mode == "lightweight":
            self.config["num_worldlines"] = min(1000, self.config.get("num_worldlines", 100000))
            self.config["max_simulation_depth"] = min(30, self.config.get("max_simulation_depth", 90))
            self.config["parallel_workers"] = min(4, self.config.get("parallel_workers", 16))
        elif self.mode == "heavy":
            self.config["num_worldlines"] = max(1000000, self.config.get("num_worldlines", 100000))
            self.config["max_simulation_depth"] = max(180, self.config.get("max_simulation_depth", 90))
            self.config["parallel_workers"] = max(32, self.config.get("parallel_workers", 16))
    
    def start(self):
        """启动引擎"""
        if self.running:
            self.logger.warning("引擎已经在运行中")
            return
        
        self.logger.info("启动Nexus引擎...")
        self.running = True
        self.stop_event.clear()
        
        # 在单独的线程中运行引擎主循环
        self.engine_thread = threading.Thread(target=self._main_loop)
        self.engine_thread.daemon = True
        self.engine_thread.start()
        
        self.logger.info("引擎已成功启动")
    
    def stop(self):
        """停止引擎"""
        if not self.running:
            self.logger.warning("引擎已经停止")
            return
        
        self.logger.info("停止Nexus引擎...")
        self.running = False
        self.stop_event.set()
        
        if self.engine_thread:
            self.engine_thread.join(timeout=30)
        
        # 清理资源
        self.world_simulator.shutdown()
        self.logger.info("引擎已成功停止")
    
    def restart(self):
        """重启引擎"""
        self.stop()
        time.sleep(2)
        self.start()
    
    def is_running(self) -> bool:
        """检查引擎是否正在运行"""
        return self.running
    
    def _main_loop(self):
        """引擎主循环"""
        try:
            self.logger.info("开始主循环...")
            
            # 初始数据收集
            self.logger.info("执行初始数据收集...")
            self.current_reality_data = self.data_collector.collect_all_data()
            processed_data = self.data_processor.process(self.current_reality_data)
            
            # 生成初始世界线种子
            self.logger.info(f"生成初始世界线种子 (数量: {self.config.get('num_worldlines', 100000)})...")
            self.active_worldlines = self.seed_generator.generate_initial_seeds(processed_data)
            
            iteration_count = 0
            
            while self.running and not self.stop_event.is_set():
                start_time = time.time()
                iteration_count += 1
                
                self.logger.info(f"===== 迭代 {iteration_count} 开始 =====")
                
                # 1. 模拟世界线演化
                self.logger.info("运行世界线模拟...")
                simulated_worldlines = self.world_simulator.simulate(
                    self.active_worldlines,
                    self.config.get("simulation_steps_per_iteration", 1)
                )
                
                # 2. 收集新的现实数据
                self.logger.info("收集最新现实数据...")
                new_reality_data = self.data_collector.collect_all_data()
                processed_reality = self.data_processor.process(new_reality_data)
                
                # 3. 比对模拟与现实
                self.logger.info("比对模拟结果与现实数据...")
                comparison_results = self.reality_comparator.compare(
                    simulated_worldlines, 
                    processed_reality
                )
                
                # 4. 筛选存活的世界线
                self.logger.info("筛选存活的世界线...")
                self.survivor_worldlines = self.survivor_selector.select(
                    simulated_worldlines, 
                    comparison_results
                )
                
                # 更新指标
                survival_rate = len(self.survivor_worldlines) / len(simulated_worldlines) * 100
                self.metrics.record_survival_rate(survival_rate)
                self.metrics.record_iteration_time(time.time() - start_time)
                
                self.logger.info(f"本轮迭代完成: {len(self.survivor_worldlines)} 个世界线存活 ({survival_rate:.4f}%)")
                
                # 5. 提取洞察
                if len(self.survivor_worldlines) > 0:
                    insights = self.insight_extractor.extract(self.survivor_worldlines)
                    self.logger.info(f"提取到 {len(insights)} 个关键洞察")
                
                # 6. 生成新的世界线种子
                self.logger.info("生成新一代世界线种子...")
                self.active_worldlines = self.seed_generator.generate_new_generation(
                    self.survivor_worldlines,
                    processed_reality,
                    self.config.get("num_worldlines", 100000)
                )
                
                # 更新当前现实数据
                self.current_reality_data = new_reality_data
                
                # 等待到下一个迭代周期
                iteration_interval = self.config.get("iteration_interval_seconds", 3600)  # 默认1小时
                self.logger.info(f"等待 {iteration_interval} 秒到下一次迭代...")
                self.stop_event.wait(iteration_interval)
                
        except Exception as e:
            self.logger.error(f"主循环出错: {str(e)}", exc_info=True)
            self.running = False
    
    def test_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        测试策略在存活世界线中的效果
        
        Args:
            strategy: 策略定义字典
            
        Returns:
            策略测试结果
        """
        if not self.survivor_worldlines:
            self.logger.warning("没有可用的存活世界线，使用随机生成的世界线进行测试")
            # 使用最近的现实数据生成临时世界线
            if self.current_reality_data:
                processed_data = self.data_processor.process(self.current_reality_data)
                temp_worldlines = self.seed_generator.generate_initial_seeds(
                    processed_data, 
                    count=1000  # 生成较少的世界线用于快速测试
                )
            else:
                self.logger.error("没有可用的现实数据，无法测试策略")
                return {
                    "success": False,
                    "error": "No reality data available",
                    "success_rate": 0.0,
                    "key_impacts": []
                }
        else:
            temp_worldlines = self.survivor_worldlines.copy()
        
        # 注入策略并评估
        self.logger.info(f"测试策略: {strategy.get('type')}")
        results = self.strategy_injector.evaluate_strategy(
            temp_worldlines,
            strategy,
            simulation_steps=30  # 模拟30天后的结果
        )
        
        return results
    
    def get_most_likely_futures(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        获取当前最可能的未来世界线
        
        Args:
            count: 返回的世界线数量
            
        Returns:
            最可能的未来世界线列表
        """
        if not self.survivor_worldlines:
            self.logger.warning("没有可用的存活世界线")
            return []
        
        # 根据生存分数排序，返回前N个
        sorted_worldlines = sorted(
            self.survivor_worldlines, 
            key=lambda x: x.get("survival_score", 0), 
            reverse=True
        )
        
        return sorted_worldlines[:count]
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取引擎运行指标"""
        return {
            "iteration_count": self.metrics.iteration_count,
            "average_survival_rate": self.metrics.average_survival_rate,
            "average_iteration_time": self.metrics.average_iteration_time,
            "active_worldlines": len(self.active_worldlines),
            "survivor_worldlines": len(self.survivor_worldlines),
            "uptime_seconds": self.metrics.uptime_seconds
        }

# 简单的启动函数（用于直接运行此模块）
def main():
    """测试运行Nexus引擎"""
    # 简单配置
    config = {
        "num_worldlines": 1000,
        "max_simulation_depth": 90,
        "parallel_workers": 4,
        "simulation_steps_per_iteration": 1,
        "iteration_interval_seconds": 60  # 为了测试，设置较短的间隔
    }
    
    engine = NexusEngine(config, mode="lightweight")
    engine.start()
    
    try:
        # 运行5分钟后停止
        time.sleep(300)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()

if __name__ == "__main__":
    main()