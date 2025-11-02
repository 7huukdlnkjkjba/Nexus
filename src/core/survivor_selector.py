#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
幸存者选择器模块
负责从模拟世界线中筛选最接近现实的世界线
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

from ..utils.logger import get_logger
from ..data.data_types import WorldLine

class SurvivorSelector:
    """幸存者选择器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化幸存者选择器
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("SurvivorSelector")
        self.config = config
        
        # 选择策略参数
        self.selection_strategy = config.get("selection_strategy", "top_percentile")  # 可选: top_percentile, fixed_count, threshold
        
        # 按百分比选择
        self.survival_percentile = config.get("survival_percentile", 0.1)  # 10%的世界线存活
        
        # 固定数量选择
        self.fixed_survivor_count = config.get("fixed_survivor_count", 10000)  # 固定存活数量
        
        # 阈值选择
        self.min_similarity_threshold = config.get("min_similarity_threshold", 0.7)  # 最低相似度阈值
        
        # 最小/最大存活数量限制
        self.min_survivors = config.get("min_survivors", 100)  # 无论如何至少保留的数量
        self.max_survivors = config.get("max_survivors", 50000)  # 最多保留的数量
        
        # 多样性保护参数
        self.enable_diversity_protection = config.get("enable_diversity_protection", True)  # 是否启用多样性保护
        self.diversity_groups = config.get("diversity_groups", 10)  # 多样性分组数量
        self.diversity_survival_rate = config.get("diversity_survival_rate", 0.15)  # 每组的存活率
        
        self.logger.info("幸存者选择器初始化完成")
        self.logger.info(f"选择策略: {self.selection_strategy}")
    
    def select(self, worldlines: List[WorldLine], 
              comparison_results: List[Dict[str, float]]) -> List[WorldLine]:
        """
        从世界线中选择幸存者
        
        Args:
            worldlines: 世界线列表
            comparison_results: 比较结果列表
            
        Returns:
            存活的世界线列表
        """
        if not worldlines or not comparison_results:
            self.logger.warning("没有世界线或比较结果需要处理")
            return []
        
        if len(worldlines) != len(comparison_results):
            self.logger.error("世界线数量与比较结果数量不匹配")
            return []
        
        self.logger.info(f"开始选择幸存者，总世界线数: {len(worldlines)}")
        
        # 为世界线分配相似度分数
        worldlines_with_scores = self._assign_scores(worldlines, comparison_results)
        
        # 移除有异常的世界线
        valid_worldlines = [w for w in worldlines_with_scores if not w.get("has_anomalies", False)]
        
        if not valid_worldlines:
            self.logger.warning("没有有效的世界线，跳过异常检查")
            valid_worldlines = worldlines_with_scores
        else:
            self.logger.info(f"移除了 {len(worldlines_with_scores) - len(valid_worldlines)} 个有异常的世界线")
        
        # 根据选择策略选择幸存者
        survivors = []
        
        if self.enable_diversity_protection and len(valid_worldlines) >= self.diversity_groups * 10:
            # 启用多样性保护的选择
            survivors = self._select_with_diversity(valid_worldlines)
        else:
            # 常规选择
            survivors = self._select_regular(valid_worldlines)
        
        # 应用数量限制
        survivors = self._apply_survivor_limits(survivors)
        
        # 按分数降序排序
        survivors.sort(key=lambda x: x.get("survival_score", 0.0), reverse=True)
        
        self.logger.info(f"选择完成，存活世界线数: {len(survivors)}")
        
        # 如果有存活者，记录最高和最低分数
        if survivors:
            max_score = survivors[0].get("survival_score", 0.0)
            min_score = survivors[-1].get("survival_score", 0.0)
            self.logger.info(f"最高分数: {max_score:.4f}, 最低分数: {min_score:.4f}")
        
        return survivors
    
    def _assign_scores(self, worldlines: List[WorldLine], 
                      comparison_results: List[Dict[str, float]]) -> List[WorldLine]:
        """
        为世界线分配相似度分数
        
        Args:
            worldlines: 世界线列表
            comparison_results: 比较结果列表
            
        Returns:
            带分数的世界线列表
        """
        # 创建ID到结果的映射
        result_map = {result["worldline_id"]: result for result in comparison_results}
        
        worldlines_with_scores = []
        
        for worldline in worldlines:
            w_id = worldline["id"]
            if w_id in result_map:
                result = result_map[w_id]
                # 复制世界线并添加分数
                scored_worldline = worldline.copy()
                scored_worldline["survival_score"] = result["similarity_score"]
                scored_worldline["economic_similarity"] = result["economic_similarity"]
                scored_worldline["political_similarity"] = result["political_similarity"]
                scored_worldline["technological_similarity"] = result["technological_similarity"]
                scored_worldline["climate_similarity"] = result["climate_similarity"]
                scored_worldline["has_anomalies"] = result["has_anomalies"]
                worldlines_with_scores.append(scored_worldline)
            else:
                self.logger.warning(f"找不到世界线 {w_id} 的比较结果")
                # 给没有结果的世界线一个默认低分数
                scored_worldline = worldline.copy()
                scored_worldline["survival_score"] = 0.0
                scored_worldline["has_anomalies"] = True
                worldlines_with_scores.append(scored_worldline)
        
        return worldlines_with_scores
    
    def _select_regular(self, worldlines: List[WorldLine]) -> List[WorldLine]:
        """
        常规选择方法
        
        Args:
            worldlines: 带分数的世界线列表
            
        Returns:
            选择的世界线列表
        """
        # 按分数降序排序
        sorted_worldlines = sorted(
            worldlines, 
            key=lambda x: x.get("survival_score", 0.0), 
            reverse=True
        )
        
        survivors = []
        
        if self.selection_strategy == "top_percentile":
            # 选择前N%的世界线
            survivor_count = max(self.min_survivors, int(len(sorted_worldlines) * self.survival_percentile))
            survivors = sorted_worldlines[:survivor_count]
            
        elif self.selection_strategy == "fixed_count":
            # 选择固定数量的世界线
            survivor_count = min(self.max_survivors, max(self.min_survivors, self.fixed_survivor_count))
            survivors = sorted_worldlines[:survivor_count]
            
        elif self.selection_strategy == "threshold":
            # 选择分数高于阈值的世界线
            survivors = [w for w in sorted_worldlines if w.get("survival_score", 0.0) >= self.min_similarity_threshold]
            
        else:
            # 默认使用top_percentile
            self.logger.warning(f"未知的选择策略: {self.selection_strategy}，使用top_percentile")
            survivor_count = max(self.min_survivors, int(len(sorted_worldlines) * self.survival_percentile))
            survivors = sorted_worldlines[:survivor_count]
        
        return survivors
    
    def _select_with_diversity(self, worldlines: List[WorldLine]) -> List[WorldLine]:
        """
        启用多样性保护的选择方法
        将世界线分组，每组选择一定比例的幸存者，以保持多样性
        
        Args:
            worldlines: 带分数的世界线列表
            
        Returns:
            选择的世界线列表
        """
        self.logger.info("使用多样性保护选择策略")
        
        # 按经济、政治、技术、气候特征对世界线进行分组
        groups = self._group_worldlines(worldlines)
        
        survivors = []
        
        for group_id, group_worldlines in groups.items():
            # 对组内世界线按分数排序
            group_worldlines.sort(key=lambda x: x.get("survival_score", 0.0), reverse=True)
            
            # 每组选择一定比例的幸存者
            group_survivor_count = max(1, int(len(group_worldlines) * self.diversity_survival_rate))
            survivors.extend(group_worldlines[:group_survivor_count])
        
        # 合并后再次排序（确保整体最高分在前）
        survivors.sort(key=lambda x: x.get("survival_score", 0.0), reverse=True)
        
        return survivors
    
    def _group_worldlines(self, worldlines: List[WorldLine]) -> Dict[int, List[WorldLine]]:
        """
        对世界线进行分组以保持多样性
        
        Args:
            worldlines: 带分数的世界线列表
            
        Returns:
            分组后的世界线字典
        """
        groups = defaultdict(list)
        
        for worldline in worldlines:
            # 创建一个特征向量用于分组
            features = self._extract_diversity_features(worldline)
            
            # 使用简单的哈希方法将特征向量映射到组ID
            group_id = self._hash_features_to_group(features)
            groups[group_id].append(worldline)
        
        return groups
    
    def _extract_diversity_features(self, worldline: WorldLine) -> List[float]:
        """
        提取用于多样性分组的特征
        
        Args:
            worldline: 世界线
            
        Returns:
            特征向量
        """
        state = worldline.get("state", {})
        
        features = []
        
        # 经济特征 - GDP增长率（这里简化处理）
        if "economic" in state and "gdp" in state["economic"]:
            gdp_values = list(state["economic"]["gdp"].values())
            if gdp_values:
                avg_gdp = np.mean(gdp_values)
                features.append(avg_gdp)
        
        # 政治特征 - 平均紧张度
        if "political" in state and "tensions" in state["political"]:
            tension_values = list(state["political"]["tensions"].values())
            if tension_values:
                avg_tension = np.mean(tension_values)
                features.append(avg_tension)
        
        # 技术特征 - 平均技术水平
        if "technological" in state:
            tech_values = [v for k, v in state["technological"].items() if isinstance(v, (int, float))]
            if tech_values:
                avg_tech = np.mean(tech_values)
                features.append(avg_tech)
        
        # 气候特征 - 全球温度
        if "climate" in state and "global_temp_increase" in state["climate"]:
            features.append(state["climate"]["global_temp_increase"])
        
        # 如果没有足够的特征，使用相似度分数
        if not features:
            features.append(worldline.get("survival_score", 0.0))
        
        return features
    
    def _hash_features_to_group(self, features: List[float]) -> int:
        """
        将特征向量哈希到组ID
        
        Args:
            features: 特征向量
            
        Returns:
            组ID
        """
        # 简单的哈希方法
        hash_value = 0
        for i, feature in enumerate(features):
            # 将特征映射到0-1区间（简化处理）
            normalized = min(1.0, max(0.0, (feature - np.min(features)) / (np.max(features) - np.min(features) + 1e-10)))
            # 分配到不同的组
            hash_value += int(normalized * (self.diversity_groups - 1)) * (self.diversity_groups ** i)
        
        # 确保在组数量范围内
        group_id = hash_value % self.diversity_groups
        
        return group_id
    
    def _apply_survivor_limits(self, survivors: List[WorldLine]) -> List[WorldLine]:
        """
        应用幸存者数量限制
        
        Args:
            survivors: 选择的世界线列表
            
        Returns:
            限制后的世界线列表
        """
        survivor_count = len(survivors)
        
        # 应用最小限制
        if survivor_count < self.min_survivors:
            self.logger.warning(f"幸存者数量 {survivor_count} 低于最小限制 {self.min_survivors}")
            # 这里不增加数量，因为我们没有更多的世界线可以选择
        
        # 应用最大限制
        elif survivor_count > self.max_survivors:
            self.logger.warning(f"幸存者数量 {survivor_count} 超过最大限制 {self.max_survivors}")
            # 保留分数最高的
            survivors = sorted(
                survivors, 
                key=lambda x: x.get("survival_score", 0.0), 
                reverse=True
            )[:self.max_survivors]
        
        return survivors

# 测试代码
if __name__ == "__main__":
    # 简单的测试配置
    config = {
        "selection_strategy": "top_percentile",
        "survival_percentile": 0.2,  # 20%的世界线存活
        "min_survivors": 1,
        "max_survivors": 100,
        "enable_diversity_protection": True,
        "diversity_groups": 5
    }
    
    # 创建选择器
    selector = SurvivorSelector(config)
    
    # 创建测试世界线
    from datetime import datetime
    import uuid
    
    worldlines = []
    comparison_results = []
    
    # 创建10个世界线，分数从0.9到0.0
    for i in range(10):
        w_id = str(uuid.uuid4())
        score = 0.9 - i * 0.1
        
        # 世界线
        worldline = {
            "id": w_id,
            "current_time": datetime.now().isoformat(),
            "state": {
                "economic": {"gdp": {"US": 25.0 + i, "China": 18.0 - i}},
                "political": {"tensions": {"region1": 50 + i * 5}},
                "technological": {"ai_progress": 70 - i * 7},
                "climate": {"global_temp_increase": 1.2 + i * 0.1}
            }
        }
        worldlines.append(worldline)
        
        # 比较结果
        result = {
            "worldline_id": w_id,
            "similarity_score": score,
            "economic_similarity": score + 0.05 * (i % 2),
            "political_similarity": score - 0.05 * (i % 2),
            "technological_similarity": score + 0.03 * (i % 3),
            "climate_similarity": score - 0.03 * (i % 3),
            "time_factor": 1.0,
            "has_anomalies": i >= 8  # 最后两个有异常
        }
        comparison_results.append(result)
    
    # 执行选择
    survivors = selector.select(worldlines, comparison_results)
    
    # 打印结果
    print(f"原始世界线数: {len(worldlines)}")
    print(f"选择的世界线数: {len(survivors)}")
    print("\n选择的世界线分数:")
    for survivor in survivors:
        print(f"分数: {survivor.get('survival_score', 0.0):.4f}")
    
    # 验证选择结果
    assert len(survivors) <= 2, "应该只选择前20%（2个）世界线"
    assert all(s.get("has_anomalies", False) == False for s in survivors), "不应该选择有异常的世界线"
    print("\n测试通过！")