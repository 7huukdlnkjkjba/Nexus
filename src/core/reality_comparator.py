#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
现实比较器模块
负责将模拟结果与现实数据进行比对，计算相似度分数
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..utils.logger import get_logger
from ..data.data_types import WorldLine, RealityData
from ..utils.metrics import calculate_rmse, calculate_mae, calculate_cosine_similarity

class RealityComparator:
    """现实比较器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化现实比较器
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("RealityComparator")
        self.config = config
        
        # 相似度计算参数
        self.weight_economic = config.get("weight_economic", 0.4)  # 经济因素权重
        self.weight_political = config.get("weight_political", 0.3)  # 政治因素权重
        self.weight_technological = config.get("weight_technological", 0.2)  # 技术因素权重
        self.weight_climate = config.get("weight_climate", 0.1)  # 气候因素权重
        
        # 时间衰减因子 - 模拟时间离现实越远，分数越低
        self.time_decay_factor = config.get("time_decay_factor", 0.95)
        
        # 异常容忍度 - 允许的最大偏差比例
        self.max_tolerable_deviation = config.get("max_tolerable_deviation", 0.3)  # 30%
        
        # 确保权重和为1
        total_weight = (self.weight_economic + self.weight_political + 
                       self.weight_technological + self.weight_climate)
        if total_weight != 1.0:
            self.logger.warning(f"权重和不为1，正在归一化: {total_weight}")
            self.weight_economic /= total_weight
            self.weight_political /= total_weight
            self.weight_technological /= total_weight
            self.weight_climate /= total_weight
        
        self.logger.info("现实比较器初始化完成")
        self.logger.info(f"权重设置: 经济={self.weight_economic}, 政治={self.weight_political}, "
                       f"技术={self.weight_technological}, 气候={self.weight_climate}")
    
    def compare(self, worldlines: List[WorldLine], 
               reality_data: RealityData) -> List[Dict[str, float]]:
        """
        比较多个世界线与现实数据
        
        Args:
            worldlines: 要比较的世界线列表
            reality_data: 现实数据
            
        Returns:
            每个世界线的比较结果列表，包含相似度分数等信息
        """
        if not worldlines:
            self.logger.warning("没有世界线需要比较")
            return []
        
        self.logger.info(f"开始比较 {len(worldlines)} 个世界线与现实数据")
        
        comparison_results = []
        for i, worldline in enumerate(worldlines):
            try:
                result = self._compare_single_worldline(worldline, reality_data)
                comparison_results.append(result)
                
                # 记录进度
                if (i + 1) % 1000 == 0:
                    self.logger.info(f"已比较 {i + 1} 个世界线")
            except Exception as e:
                self.logger.error(f"比较世界线 {worldline.get('id')} 失败: {str(e)}")
                # 为失败的比较提供一个默认的低分数结果
                comparison_results.append({
                    "worldline_id": worldline.get('id'),
                    "similarity_score": 0.0,
                    "economic_similarity": 0.0,
                    "political_similarity": 0.0,
                    "technological_similarity": 0.0,
                    "climate_similarity": 0.0,
                    "time_factor": 1.0,
                    "has_anomalies": True
                })
        
        self.logger.info(f"比较完成，共 {len(comparison_results)} 个结果")
        return comparison_results
    
    def _compare_single_worldline(self, worldline: WorldLine, 
                                 reality_data: RealityData) -> Dict[str, float]:
        """
        比较单个世界线与现实数据
        
        Args:
            worldline: 要比较的世界线
            reality_data: 现实数据
            
        Returns:
            比较结果字典
        """
        # 获取世界线的当前状态
        worldline_state = worldline["state"]
        
        # 计算时间因子（考虑模拟时间与现实的差异）
        time_factor = self._calculate_time_factor(worldline)
        
        # 计算各领域相似度
        economic_similarity = self._compare_economic_data(
            worldline_state.get("economic", {}),
            reality_data.get("economic", {})
        )
        
        political_similarity = self._compare_political_data(
            worldline_state.get("political", {}),
            reality_data.get("political", {})
        )
        
        technological_similarity = self._compare_technological_data(
            worldline_state.get("technological", {}),
            reality_data.get("technological", {})
        )
        
        climate_similarity = self._compare_climate_data(
            worldline_state.get("climate", {}),
            reality_data.get("climate", {})
        )
        
        # 计算加权综合相似度分数
        weighted_score = (
            self.weight_economic * economic_similarity +
            self.weight_political * political_similarity +
            self.weight_technological * technological_similarity +
            self.weight_climate * climate_similarity
        )
        
        # 应用时间衰减
        similarity_score = weighted_score * time_factor
        
        # 检查是否存在异常偏差
        has_anomalies = self._check_for_anomalies(
            economic_similarity,
            political_similarity,
            technological_similarity,
            climate_similarity
        )
        
        return {
            "worldline_id": worldline["id"],
            "similarity_score": similarity_score,
            "economic_similarity": economic_similarity,
            "political_similarity": political_similarity,
            "technological_similarity": technological_similarity,
            "climate_similarity": climate_similarity,
            "time_factor": time_factor,
            "has_anomalies": has_anomalies
        }
    
    def _calculate_time_factor(self, worldline: WorldLine) -> float:
        """
        计算时间因子，模拟时间离现在越远，因子越小
        
        Args:
            worldline: 世界线
            
        Returns:
            时间因子（0-1之间）
        """
        try:
            # 获取世界线当前时间
            worldline_time = datetime.fromisoformat(worldline["current_time"])
            now = datetime.now()
            
            # 计算时间差（天数）
            days_diff = abs((worldline_time - now).days)
            
            # 应用衰减因子，每天衰减一点
            time_factor = self.time_decay_factor ** days_diff
            
            # 确保因子在合理范围内
            time_factor = max(0.1, min(1.0, time_factor))
            
            return time_factor
        except Exception as e:
            self.logger.error(f"计算时间因子失败: {str(e)}")
            return 1.0  # 默认返回1.0
    
    def _compare_economic_data(self, worldline_economic: Dict[str, Any],
                              reality_economic: Dict[str, Any]) -> float:
        """比较经济数据"""
        if not worldline_economic or not reality_economic:
            self.logger.warning("经济数据不完整，无法进行比较")
            return 0.5  # 返回中等相似度
        
        try:
            similarities = []
            
            # 比较GDP
            if "gdp" in worldline_economic and "gdp" in reality_economic:
                gdp_similarity = self._compare_economic_indicator(
                    worldline_economic["gdp"],
                    reality_economic["gdp"]
                )
                similarities.append(gdp_similarity)
            
            # 比较通货膨胀率
            if "inflation" in worldline_economic and "inflation" in reality_economic:
                inflation_similarity = self._compare_economic_indicator(
                    worldline_economic["inflation"],
                    reality_economic["inflation"]
                )
                similarities.append(inflation_similarity)
            
            # 比较失业率
            if "unemployment" in worldline_economic and "unemployment" in reality_economic:
                unemployment_similarity = self._compare_economic_indicator(
                    worldline_economic["unemployment"],
                    reality_economic["unemployment"]
                )
                similarities.append(unemployment_similarity)
            
            # 返回平均相似度
            return np.mean(similarities) if similarities else 0.5
        
        except Exception as e:
            self.logger.error(f"比较经济数据失败: {str(e)}")
            return 0.5
    
    def _compare_economic_indicator(self, worldline_data: Dict[str, float],
                                   reality_data: Dict[str, float]) -> float:
        """比较单个经济指标"""
        if not worldline_data or not reality_data:
            return 0.5
        
        # 获取共同的国家/地区
        common_countries = set(worldline_data.keys()) & set(reality_data.keys())
        
        if not common_countries:
            return 0.3  # 没有共同数据，返回较低相似度
        
        # 准备数据数组
        worldline_values = []
        reality_values = []
        
        for country in common_countries:
            worldline_values.append(worldline_data[country])
            reality_values.append(reality_data[country])
        
        # 计算相对误差而不是绝对误差（对经济数据更合适）
        relative_errors = []
        for wv, rv in zip(worldline_values, reality_values):
            if rv != 0:
                relative_error = abs(wv - rv) / abs(rv)
                # 将误差转换为相似度（误差越小，相似度越高）
                similarity = max(0, 1 - relative_error / self.max_tolerable_deviation)
                relative_errors.append(similarity)
            else:
                # 如果现实值为0，则直接比较
                similarity = 1.0 if abs(wv - rv) < 0.001 else 0.0
                relative_errors.append(similarity)
        
        return np.mean(relative_errors)
    
    def _compare_political_data(self, worldline_political: Dict[str, Any],
                               reality_political: Dict[str, Any]) -> float:
        """比较政治数据"""
        if not worldline_political or not reality_political:
            self.logger.warning("政治数据不完整，无法进行比较")
            return 0.5
        
        try:
            similarities = []
            
            # 比较紧张度
            if "tensions" in worldline_political and "tensions" in reality_political:
                tension_similarity = self._compare_tensions(
                    worldline_political["tensions"],
                    reality_political["tensions"]
                )
                similarities.append(tension_similarity)
            
            # 比较联盟（使用集合相似度）
            if "alliances" in worldline_political and "alliances" in reality_political:
                alliance_similarity = self._compare_alliances(
                    worldline_political["alliances"],
                    reality_political["alliances"]
                )
                similarities.append(alliance_similarity)
            
            return np.mean(similarities) if similarities else 0.5
        
        except Exception as e:
            self.logger.error(f"比较政治数据失败: {str(e)}")
            return 0.5
    
    def _compare_tensions(self, worldline_tensions: Dict[str, float],
                         reality_tensions: Dict[str, float]) -> float:
        """比较紧张度"""
        common_regions = set(worldline_tensions.keys()) & set(reality_tensions.keys())
        
        if not common_regions:
            return 0.3
        
        differences = []
        for region in common_regions:
            w_tension = worldline_tensions[region]
            r_tension = reality_tensions[region]
            # 紧张度差异（0-100）
            norm_diff = abs(w_tension - r_tension) / 100.0
            similarity = max(0, 1 - norm_diff / self.max_tolerable_deviation)
            differences.append(similarity)
        
        return np.mean(differences)
    
    def _compare_alliances(self, worldline_alliances: List[str],
                          reality_alliances: List[str]) -> float:
        """比较联盟（使用Jaccard相似度）"""
        w_set = set(worldline_alliances)
        r_set = set(reality_alliances)
        
        if not w_set and not r_set:
            return 1.0
        
        intersection = len(w_set & r_set)
        union = len(w_set | r_set)
        
        return intersection / union if union > 0 else 0.0
    
    def _compare_technological_data(self, worldline_tech: Dict[str, Any],
                                   reality_tech: Dict[str, Any]) -> float:
        """比较技术数据"""
        if not worldline_tech or not reality_tech:
            self.logger.warning("技术数据不完整，无法进行比较")
            return 0.5
        
        try:
            # 提取技术指标值
            w_values = []
            r_values = []
            
            for key in worldline_tech:
                if key in reality_tech and isinstance(worldline_tech[key], (int, float)):
                    w_values.append(worldline_tech[key])
                    r_values.append(reality_tech[key])
            
            if not w_values:
                return 0.4
            
            # 计算相对误差
            similarities = []
            for wv, rv in zip(w_values, r_values):
                if rv != 0:
                    rel_diff = abs(wv - rv) / rv
                    similarity = max(0, 1 - rel_diff / self.max_tolerable_deviation)
                    similarities.append(similarity)
                else:
                    similarities.append(1.0 if abs(wv) < 0.001 else 0.0)
            
            return np.mean(similarities)
        
        except Exception as e:
            self.logger.error(f"比较技术数据失败: {str(e)}")
            return 0.5
    
    def _compare_climate_data(self, worldline_climate: Dict[str, Any],
                             reality_climate: Dict[str, Any]) -> float:
        """比较气候数据"""
        if not worldline_climate or not reality_climate:
            self.logger.warning("气候数据不完整，无法进行比较")
            return 0.5
        
        try:
            # 提取气候指标值
            w_values = []
            r_values = []
            
            for key in worldline_climate:
                if key in reality_climate and isinstance(worldline_climate[key], (int, float)):
                    w_values.append(worldline_climate[key])
                    r_values.append(reality_climate[key])
            
            if not w_values:
                return 0.4
            
            # 计算相对误差（气候数据的变化通常很小）
            similarities = []
            for wv, rv in zip(w_values, r_values):
                if rv != 0:
                    # 气候数据使用较小的偏差容忍度
                    climate_tolerance = self.max_tolerable_deviation * 0.5  # 气候变化更缓慢
                    rel_diff = abs(wv - rv) / rv
                    similarity = max(0, 1 - rel_diff / climate_tolerance)
                    similarities.append(similarity)
                else:
                    similarities.append(1.0 if abs(wv) < 0.001 else 0.0)
            
            return np.mean(similarities)
        
        except Exception as e:
            self.logger.error(f"比较气候数据失败: {str(e)}")
            return 0.5
    
    def _check_for_anomalies(self, economic_similarity: float,
                            political_similarity: float,
                            technological_similarity: float,
                            climate_similarity: float) -> bool:
        """
        检查是否存在异常偏差
        
        任何一个领域的相似度低于阈值都被认为是异常
        """
        anomaly_threshold = 0.1  # 10%的相似度作为异常阈值
        
        return (
            economic_similarity < anomaly_threshold or
            political_similarity < anomaly_threshold or
            technological_similarity < anomaly_threshold or
            climate_similarity < anomaly_threshold
        )

# 测试代码
if __name__ == "__main__":
    # 简单的测试配置
    config = {
        "weight_economic": 0.4,
        "weight_political": 0.3,
        "weight_technological": 0.2,
        "weight_climate": 0.1,
        "time_decay_factor": 0.95,
        "max_tolerable_deviation": 0.3
    }
    
    # 创建比较器
    comparator = RealityComparator(config)
    
    # 模拟世界线和现实数据
    from datetime import datetime
    import uuid
    
    # 非常接近现实的世界线
    close_worldline = {
        "id": str(uuid.uuid4()),
        "current_time": datetime.now().isoformat(),
        "state": {
            "economic": {
                "gdp": {"US": 25.5, "China": 18.0, "Japan": 4.2},
                "inflation": {"US": 3.8, "EU": 4.2, "UK": 6.6},
                "unemployment": {"US": 3.9, "EU": 6.2, "Japan": 2.4}
            },
            "political": {
                "alliances": ["NATO", "EU", "ASEAN"],
                "tensions": {"Taiwan_Strait": 66, "Ukraine": 79, "Middle_East": 76}
            },
            "technological": {
                "ai_progress": 71,
                "renewable_energy": 44,
                "quantum_computing": 31
            },
            "climate": {
                "global_temp_increase": 1.21,
                "co2_levels": 420,
                "extreme_events": 43
            }
        }
    }
    
    # 与现实相差较大的世界线
    far_worldline = {
        "id": str(uuid.uuid4()),
        "current_time": datetime.now().isoformat(),
        "state": {
            "economic": {
                "gdp": {"US": 20.0, "China": 25.0, "Japan": 5.0},
                "inflation": {"US": 10.0, "EU": 12.0, "UK": 15.0},
                "unemployment": {"US": 10.0, "EU": 15.0, "Japan": 8.0}
            },
            "political": {
                "alliances": ["New_Alliance"],
                "tensions": {"Taiwan_Strait": 95, "Ukraine": 98, "Middle_East": 100}
            },
            "technological": {
                "ai_progress": 95,
                "renewable_energy": 80,
                "quantum_computing": 70
            },
            "climate": {
                "global_temp_increase": 2.0,
                "co2_levels": 500,
                "extreme_events": 100
            }
        }
    }
    
    # 现实数据
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
    
    # 进行比较
    results = comparator.compare([close_worldline, far_worldline], reality_data)
    
    # 打印结果
    for result in results:
        print(f"世界线 {result['worldline_id'][:8]} 相似度分数: {result['similarity_score']:.4f}")
        print(f"  经济相似度: {result['economic_similarity']:.4f}")
        print(f"  政治相似度: {result['political_similarity']:.4f}")
        print(f"  技术相似度: {result['technological_similarity']:.4f}")
        print(f"  气候相似度: {result['climate_similarity']:.4f}")
        print(f"  异常检测: {'有异常' if result['has_anomalies'] else '正常'}")
        print()
    
    # 验证我们的比较器能正确区分接近和远离现实的世界线
    assert results[0]['similarity_score'] > results[1]['similarity_score'], "比较器逻辑错误"