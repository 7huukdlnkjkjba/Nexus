#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phantom Crawler Nexus - 测试脚本

用于快速测试模拟系统的核心功能，运行小规模的模拟测试。
"""

import os
import sys
import time
import json
import logging
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("nexus.test")


def run_small_test_simulation():
    """
    运行小规模测试模拟
    """
    try:
        # 创建测试配置
        test_config = {
            "simulation": {
                "seed": 42,  # 固定种子以确保结果可重现
                "initial_year": 2023,
                "target_year": 2025,  # 只模拟2年
                "max_timelines": 5,  # 限制世界线数量
                "timeline_pruning_threshold": 0.05,
                "decision_points_per_year": 2,  # 每年2个决策点
                "save_interval": 1,
                "output_directory": f"test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            },
            "models": {
                "economic": {
                    "initial_gdp": 100.0,
                    "gdp_growth_volatility": 0.01,  # 降低波动性
                    "inflation_base_rate": 0.02,
                    "technology_contribution": 0.3,
                    "globalization_factor": 0.7
                },
                "political": {
                    "base_stability": 0.75,  # 增加稳定性以减少复杂性
                    "conflict_impact": 0.15,
                    "democracy_index": 0.65
                },
                "technology": {
                    "base_research_intensity": 0.02,
                    "breakthrough_probability": 0.03,  # 降低突破概率
                    "technology_spillover": 0.7
                },
                "climate": {
                    "base_emission_rate": 50.0,
                    "temperature_sensitivity": 0.7,
                    "carbon_budget": 1000.0,
                    "climate_policy_impact": 0.5
                },
                "crisis_response": {
                    "max_country_responses_per_event": 3,  # 减少响应数量
                    "max_organization_responses_per_event": 2,
                    "base_response_effectiveness": 0.6
                },
                "evaluator": {
                    "survival_weight": 0.6,
                    "value_weight": 0.4,
                    "domain_weights": {
                        "economic": 0.25,
                        "political": 0.25,
                        "technology": 0.25,
                        "climate": 0.25
                    }
                },
                "event_system": {
                    "base_event_probability": 0.1,  # 降低事件概率
                    "max_events_per_period": 2,
                    "severity_distribution": {
                        "minor": 0.6,  # 增加轻微事件比例
                        "moderate": 0.3,
                        "major": 0.08,
                        "catastrophic": 0.02  # 减少灾难性事件
                    }
                },
                "timeline_manager": {
                    "cache_size": 100,
                    "prune_frequency": 2
                }
            }
        }
        
        # 保存测试配置
        config_path = "test_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(test_config, f, indent=2, ensure_ascii=False)
        
        logger.info("测试配置已创建: test_config.json")
        
        # 运行主程序
        from main import load_configuration, create_output_directory, run_simulation
        
        print("\n===== 开始测试模拟 =====")
        print("这是一个小规模测试，将模拟少量世界线在短时间内的演化。")
        print("========================\n")
        
        # 加载配置
        config = load_configuration(config_path)
        
        # 创建输出目录
        output_dir = create_output_directory(config)
        
        # 运行模拟
        start_time = time.time()
        run_simulation(config, output_dir)
        end_time = time.time()
        
        print(f"\n测试模拟完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"测试结果保存在: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"测试模拟失败: {str(e)}", exc_info=True)
        print(f"\n测试失败: {str(e)}")
        return 1


def test_individual_components():
    """
    测试各个核心组件
    """
    print("\n===== 开始组件测试 =====")
    
    try:
        # 测试世界线类
        from src.data.data_types import WorldLine
        print("测试 WorldLine 类...")
        world_line = WorldLine(
            id="test_world_001",
            current_time=2023,
            seed=42,
            generation=0,
            birth_time=2023,
            survival_probability=0.8,
            value_score=0.75
        )
        print(f"  ✓ WorldLine 创建成功: {world_line.id}")
        
        # 测试时间线管理器
        from src.core.timeline_manager import TimelineManager
        print("测试 TimelineManager 类...")
        timeline_manager = TimelineManager(seed=42)
        timeline_manager.add_worldline(world_line)
        timelines = timeline_manager.get_worldlines()
        print(f"  ✓ TimelineManager 添加/获取成功，当前世界线数: {len(timelines)}")
        
        # 测试评估器
        from src.models.evaluator import WorldLineEvaluator
        print("测试 WorldLineEvaluator 类...")
        evaluator = WorldLineEvaluator()
        scores = evaluator.evaluate_worldline(world_line)
        print(f"  ✓ WorldLineEvaluator 评估成功: 生存概率={scores['survival_probability']}, 价值分数={scores['value_score']}")
        
        # 测试事件系统
        from src.core.event_system import EventSystem
        print("测试 EventSystem 类...")
        event_system = EventSystem(seed=42)
        events = event_system.generate_events(current_time=2023, timeline_count=3)
        print(f"  ✓ EventSystem 事件生成成功，生成事件数: {len(events)}")
        
        print("\n所有组件测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"组件测试失败: {str(e)}", exc_info=True)
        print(f"\n组件测试失败: {str(e)}")
        return False


def main():
    """
    主测试函数
    """
    print("""
    ====================================================
             Phantom Crawler Nexus - 测试脚本
    ====================================================
    
    1. 运行小规模模拟测试
    2. 测试各个核心组件
    ====================================================
    """)
    
    # 先测试组件
    print("\n1. 正在测试各个核心组件...")
    components_ok = test_individual_components()
    
    if not components_ok:
        print("\n组件测试失败，建议修复后再运行完整测试。")
        return 1
    
    # 再运行小规模模拟
    print("\n2. 正在运行小规模模拟测试...")
    return run_small_test_simulation()


if __name__ == "__main__":
    sys.exit(main())