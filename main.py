#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phantom Crawler Nexus - 世界线模拟器

主程序入口，整合世界线模拟系统的所有组件，提供完整的模拟运行界面。
"""

import os
import sys
import time
import json
import logging
import argparse
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'nexus_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("nexus.main")

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入核心模块
from src.core.simulation_coordinator import SimulationCoordinator
from src.core.timeline_manager import TimelineManager
from src.core.event_system import EventSystem

# 导入数据类型
from src.data.data_types import (
    WorldLine, RealityData, DataCatalog
)
from src.utils.data_types import WorldEvent as Event, EventType

# 导入模型
from src.models.evaluator import WorldLineEvaluator
from src.models.economic_model import EconomicModel
from src.models.political_model import PoliticalModel
from src.models.technology_model import TechnologyModel
from src.models.climate_model import ClimateModel
from src.models.crisis_response_model import CrisisResponseModel

def load_configuration(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载模拟配置
    
    Args:
        config_path: 配置文件路径，如果为 None 则使用默认配置
        
    Returns:
        配置字典
    """
    # 默认配置
    default_config = {
        "simulation": {
            "seed": int(time.time()),
            "initial_year": 2023,
            "target_year": 2100,
            "max_timelines": 20,
            "timeline_pruning_threshold": 0.1,
            "decision_points_per_year": 4,
            "save_interval": 5,  # 每5年保存一次状态
            "output_directory": f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        },
        "models": {
            "economic": {
                "initial_gdp": 100.0,
                "gdp_growth_volatility": 0.02,
                "inflation_base_rate": 0.02,
                "technology_contribution": 0.3,
                "globalization_factor": 0.7
            },
            "political": {
                "base_stability": 0.7,
                "conflict_impact": 0.2,
                "democracy_index": 0.65
            },
            "technology": {
                "base_research_intensity": 0.025,
                "breakthrough_probability": 0.05,
                "technology_spillover": 0.7
            },
            "climate": {
                "base_emission_rate": 50.0,
                "temperature_sensitivity": 0.7,
                "carbon_budget": 1000.0,
                "climate_policy_impact": 0.5
            },
            "crisis_response": {
                "max_country_responses_per_event": 5,
                "max_organization_responses_per_event": 3,
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
                "base_event_probability": 0.15,
                "max_events_per_period": 3,
                "severity_distribution": {
                    "minor": 0.5,
                    "moderate": 0.3,
                    "major": 0.15,
                    "catastrophic": 0.05
                }
            },
            "timeline_manager": {
                "cache_size": 1000,
                "prune_frequency": 5
            }
        },
        "visualization": {
            "enable": True,
            "plot_types": ["timeline", "heatmap", "network"],
            "resolution": "high"
        }
    }
    
    # 如果提供了配置文件路径，读取并合并配置
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                
            # 合并配置
            default_config = _merge_configs(default_config, user_config)
            logger.info(f"已从 {config_path} 加载自定义配置")
            
        except Exception as e:
            logger.error(f"读取配置文件时出错: {str(e)}")
            logger.info("使用默认配置")
    else:
        logger.info("使用默认配置")
    
    return default_config


def _merge_configs(base_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并配置字典
    
    Args:
        base_config: 基础配置
        user_config: 用户配置
        
    Returns:
        合并后的配置
    """
    result = base_config.copy()
    
    for key, value in user_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def create_output_directory(config: Dict[str, Any]) -> str:
    """
    创建输出目录
    
    Args:
        config: 配置字典
        
    Returns:
        输出目录路径
    """
    output_dir = config["simulation"]["output_directory"]
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子目录
    subdirs = ["timelines", "events", "metrics", "visualizations", "states"]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # 保存使用的配置到输出目录
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    logger.info(f"输出目录创建完成: {output_dir}")
    return output_dir


def prepare_simulation(config: Dict[str, Any]) -> SimulationCoordinator:
    """
    准备模拟环境并创建协调器
    
    Args:
        config: 配置字典
        
    Returns:
        模拟协调器实例
    """
    logger.info("开始准备模拟环境...")
    
    # 构建协调器配置
    coordinator_config = {
        "seed": config["simulation"]["seed"],
        "initial_year": config["simulation"]["initial_year"],
        "target_year": config["simulation"]["target_year"],
        "max_timelines": config["simulation"]["max_timelines"],
        "timeline_pruning_threshold": config["simulation"]["timeline_pruning_threshold"],
        "decision_points_per_year": config["simulation"]["decision_points_per_year"],
        
        # 各模型配置
        "economic_model": config["models"]["economic"],
        "political_model": config["models"]["political"],
        "technology_model": config["models"]["technology"],
        "climate_model": config["models"]["climate"],
        "crisis_response_model": config["models"]["crisis_response"],
        "evaluator": config["models"]["evaluator"],
        "event_system": config["models"]["event_system"],
        "timeline_manager": config["models"]["timeline_manager"]
    }
    
    # 创建协调器
    coordinator = SimulationCoordinator(coordinator_config)
    
    logger.info("模拟环境准备完成")
    return coordinator

def save_simulation_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    保存模拟结果
    
    Args:
        results: 模拟结果字典
        output_dir: 输出目录路径
    """
    logger.info("开始保存模拟结果...")
    
    # 保存世界线数据
    timelines_dir = os.path.join(output_dir, "timelines")
    for timeline in results["timelines"]:
        timeline_data = {
            "id": timeline.id,
            "current_time": timeline.current_time,
            "seed": timeline.seed,
            "generation": timeline.generation,
            "birth_time": timeline.birth_time,
            "survival_probability": timeline.survival_probability,
            "value_score": timeline.value_score,
            "economic_state": timeline.economic_state,
            "political_state": timeline.political_state,
            "technology_state": timeline.technology_state,
            "climate_state": timeline.climate_state,
            "timeline_history": timeline.history
        }
        
        # 保存到 JSON 文件
        filename = f"timeline_{timeline.id.replace('/', '_')}.json"
        with open(os.path.join(timelines_dir, filename), "w", encoding="utf-8") as f:
            json.dump(timeline_data, f, indent=2, ensure_ascii=False, default=str)
    
    # 保存事件数据
    events_dir = os.path.join(output_dir, "events")
    all_events = []
    for timeline in results["timelines"]:
        for event in timeline.events:
            event_dict = {
                "id": event.id,
                "timeline_id": timeline.id,
                "type": str(event.type),
                "subtype": event.subtype,
                "description": event.description,
                "severity": str(event.severity),
                "region": event.region,
                "scope": event.scope,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "probability": event.probability,
                "metadata": event.metadata
            }
            all_events.append(event_dict)
    
    with open(os.path.join(events_dir, "all_events.json"), "w", encoding="utf-8") as f:
        json.dump(all_events, f, indent=2, ensure_ascii=False, default=str)
    
    # 保存评估结果
    metrics_dir = os.path.join(output_dir, "metrics")
    with open(os.path.join(metrics_dir, "timeline_evaluations.json"), "w", encoding="utf-8") as f:
        json.dump(results["timeline_evaluations"], f, indent=2, ensure_ascii=False)
    
    # 保存统计信息
    statistics = {
        "event_statistics": results["event_statistics"],
        "response_statistics": results["response_statistics"],
        "execution_times": results["execution_times"],
        "simulation_parameters": results["simulation_parameters"],
        "timeline_count": len(results["timelines"])
    }
    
    with open(os.path.join(metrics_dir, "statistics.json"), "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"模拟结果保存完成，共 {len(results['timelines'])} 个世界线")


def generate_summary_report(results: Dict[str, Any], output_dir: str) -> None:
    """
    生成摘要报告
    
    Args:
        results: 模拟结果字典
        output_dir: 输出目录路径
    """
    report_path = os.path.join(output_dir, "simulation_summary.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phantom Crawler Nexus 模拟摘要\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 模拟参数
        f.write("## 模拟参数\n\n")
        params = results["simulation_parameters"]
        f.write(f"- 初始年份: {params['initial_year']}\n")
        f.write(f"- 目标年份: {params['target_year']}\n")
        f.write(f"- 随机种子: {params['seed']}\n")
        f.write(f"- 最大世界线数: {params['max_timelines']}\n")
        f.write(f"- 每年决策点: {params['decision_points_per_year']}\n\n")
        
        # 世界线统计
        f.write("## 世界线统计\n\n")
        f.write(f"总世界线数: {len(results['timelines'])}\n\n")
        
        # 按生存概率排序的前5个世界线
        timelines = results['timelines']
        timelines.sort(key=lambda t: t.survival_probability, reverse=True)
        
        f.write("### 生存概率最高的5个世界线\n\n")
        for i, timeline in enumerate(timelines[:5]):
            f.write(f"#### 世界线 {i+1}: {timeline.id}\n\n")
            f.write(f"- 生存概率: {timeline.survival_probability:.4f}\n")
            f.write(f"- 价值分数: {timeline.value_score:.4f}\n")
            f.write(f"- 代数: {timeline.generation}\n")
            f.write(f"- 诞生年份: {timeline.birth_time}\n\n")
            
            # 领域状态摘要
            if hasattr(timeline, 'economic_state') and timeline.economic_state:
                gdp_growth = timeline.economic_state.get("global_gdp_growth", "N/A")
                f.write(f"  - GDP增长率: {gdp_growth}\n")
            if hasattr(timeline, 'climate_state') and timeline.climate_state:
                temp_increase = timeline.climate_state.get("global_temperature_increase", "N/A")
                f.write(f"  - 温度升高: {temp_increase}\n\n")
        
        # 按价值分数排序的前5个世界线
        timelines.sort(key=lambda t: t.value_score, reverse=True)
        
        f.write("### 价值分数最高的5个世界线\n\n")
        for i, timeline in enumerate(timelines[:5]):
            f.write(f"#### 世界线 {i+1}: {timeline.id}\n\n")
            f.write(f"- 生存概率: {timeline.survival_probability:.4f}\n")
            f.write(f"- 价值分数: {timeline.value_score:.4f}\n")
            f.write(f"- 代数: {timeline.generation}\n")
            f.write(f"- 诞生年份: {timeline.birth_time}\n\n")
        
        # 事件统计
        f.write("## 事件统计\n\n")
        event_stats = results["event_statistics"]
        
        if "events_by_type" in event_stats:
            f.write("### 事件类型分布\n\n")
            for event_type, count in event_stats["events_by_type"].items():
                f.write(f"- {event_type}: {count}\n")
            f.write("\n")
        
        if "events_by_severity" in event_stats:
            f.write("### 事件严重程度分布\n\n")
            for severity, count in event_stats["events_by_severity"].items():
                f.write(f"- {severity}: {count}\n")
            f.write("\n")
        
        # 执行时间
        f.write("## 执行时间\n\n")
        exec_times = results["execution_times"]
        for phase, duration in exec_times.items():
            f.write(f"- {phase}: {duration:.2f} 秒\n")
    
    logger.info(f"摘要报告已生成: {report_path}")


def run_simulation(config: Dict[str, Any], output_dir: str) -> None:
    """
    运行完整的模拟流程
    
    Args:
        config: 配置字典
        output_dir: 输出目录路径
    """
    try:
        # 创建协调器
        coordinator = prepare_simulation(config)
        
        # 运行模拟
        print("\n===== 开始世界线模拟 =====")
        print(f"时间范围: {config['simulation']['initial_year']} -> {config['simulation']['target_year']}")
        print(f"最大世界线数: {config['simulation']['max_timelines']}")
        print(f"随机种子: {config['simulation']['seed']}")
        print("==========================\n")
        
        start_time = time.time()
        
        # 保存间隔检查
        save_interval = config["simulation"]["save_interval"]
        save_counter = 0
        
        # 运行模拟
        coordinator.run_simulation()
        
        # 获取结果
        results = coordinator.get_simulation_results()
        
        total_time = time.time() - start_time
        results["execution_times"]["total_run"] = total_time
        
        print(f"\n===== 模拟完成 =====")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"生成世界线数: {len(results['timelines'])}")
        print(f"===================\n")
        
        # 保存结果
        save_simulation_results(results, output_dir)
        
        # 生成报告
        generate_summary_report(results, output_dir)
        
    except KeyboardInterrupt:
        logger.info("模拟被用户中断")
        print("\n模拟被用户中断")
    except Exception as e:
        logger.error(f"模拟过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\n模拟过程中发生错误: {str(e)}")
        raise


def main() -> None:
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Phantom Crawler Nexus - 世界线模拟器")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--initial-year", type=int, help="初始年份")
    parser.add_argument("--target-year", type=int, help="目标年份")
    parser.add_argument("--output", type=str, help="输出目录路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    # 设置调试级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 加载配置
    config = load_configuration(args.config)
    
    # 应用命令行参数覆盖配置
    if args.seed is not None:
        config["simulation"]["seed"] = args.seed
    if args.initial_year is not None:
        config["simulation"]["initial_year"] = args.initial_year
    if args.target_year is not None:
        config["simulation"]["target_year"] = args.target_year
    if args.output is not None:
        config["simulation"]["output_directory"] = args.output
    
    # 创建输出目录
    output_dir = create_output_directory(config)
    
    # 打印欢迎信息
    print("""
    ====================================================
             Phantom Crawler Nexus - 世界线模拟器
    ====================================================
    
    探索平行世界的可能性，模拟全球系统的演化路径
    
    支持研究领域: 经济学、政治学、技术发展、气候变化、危机响应
    ====================================================
    """)
    
    # 运行模拟
    run_simulation(config, output_dir)
    
    # 完成信息
    print("\n" + "="*56)
    print("模拟完成！所有结果已保存到输出目录。")
    print(f"结果目录: {output_dir}")
    print(f"摘要报告: {os.path.join(output_dir, 'simulation_summary.md')}")
    print("="*56)

if __name__ == "__main__":
    main()