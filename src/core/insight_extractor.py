#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
洞察提取器 (Insight Extractor)
从存活世界线中提取关键洞察，包括模式识别、转折点检测、风险信号提取等
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class InsightExtractor:
    """
    洞察提取器类，负责从模拟结果中提取有价值的洞察
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化洞察提取器
        
        Args:
            config: 配置字典，包含洞察提取器的参数
        """
        # 默认配置
        self.config = {
            'pattern_detection_threshold': 0.7,
            'turning_point_sensitivity': 0.05,
            'risk_threshold': 0.8,
            'opportunity_threshold': 0.7,
            'confidence_window': 10,
            'min_pattern_length': 5,
            'max_pattern_length': 30
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 初始化模式库
        self.patterns = {
            'trend': self._detect_trend,
            'cycle': self._detect_cycle,
            'sudden_change': self._detect_sudden_change,
            'convergence': self._detect_convergence,
            'divergence': self._detect_divergence
        }
        
        logger.info("InsightExtractor initialized with %d pattern detectors", len(self.patterns))
    
    def extract_insights(self, world_lines: List[Dict[str, Any]], 
                        reality_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        从多个存活世界线中提取综合洞察
        
        Args:
            world_lines: 存活的世界线列表
            reality_data: 可选的现实数据，用于校准
            
        Returns:
            包含各种洞察的字典
        """
        insights = {
            'patterns': [],
            'turning_points': [],
            'risks': [],
            'opportunities': [],
            'confidence_scores': {},
            'key_metrics': {}
        }
        
        if not world_lines:
            logger.warning("No world lines provided for insight extraction")
            return insights
        
        # 1. 识别模式
        insights['patterns'] = self._identify_patterns(world_lines)
        
        # 2. 检测转折点
        insights['turning_points'] = self._detect_turning_points(world_lines)
        
        # 3. 提取风险信号
        insights['risks'] = self._extract_risk_signals(world_lines)
        
        # 4. 识别机会窗口
        insights['opportunities'] = self._identify_opportunity_windows(world_lines)
        
        # 5. 计算置信度
        insights['confidence_scores'] = self._calculate_confidence_scores(world_lines, reality_data)
        
        # 6. 汇总关键指标
        insights['key_metrics'] = self._summarize_key_metrics(world_lines)
        
        logger.info(f"Extracted insights: {len(insights['patterns'])} patterns, "
                   f"{len(insights['turning_points'])} turning points, "
                   f"{len(insights['risks'])} risks, "
                   f"{len(insights['opportunities'])} opportunities")
        
        return insights
    
    def _identify_patterns(self, world_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        识别世界线中的模式
        
        Args:
            world_lines: 世界线列表
            
        Returns:
            识别出的模式列表
        """
        patterns = []
        
        for world_line in world_lines:
            world_line_id = world_line.get('id', 'unknown')
            
            for domain in world_line.get('domains', {}):
                if 'history' not in world_line['domains'][domain]:
                    continue
                
                history = world_line['domains'][domain]['history']
                if len(history) < self.config['min_pattern_length']:
                    continue
                
                # 应用各种模式检测器
                for pattern_type, detector_func in self.patterns.items():
                    detected_patterns = detector_func(history, domain, world_line_id)
                    patterns.extend(detected_patterns)
        
        # 合并相似模式
        return self._merge_similar_patterns(patterns)
    
    def _detect_trend(self, history: List[float], domain: str, 
                     world_line_id: str) -> List[Dict[str, Any]]:
        """
        检测趋势模式
        
        Args:
            history: 历史数据序列
            domain: 领域名称
            world_line_id: 世界线ID
            
        Returns:
            检测到的趋势模式列表
        """
        trends = []
        window_size = max(3, int(len(history) * 0.1))
        
        # 使用移动平均线检测趋势
        if len(history) >= window_size * 2:
            # 计算斜率
            x = np.arange(len(history))
            slope = np.polyfit(x, history, 1)[0]
            
            # 计算R²值评估趋势强度
            y_pred = np.polyval(np.polyfit(x, history, 1), x)
            r_squared = 1 - np.sum((history - y_pred)**2) / np.sum((history - np.mean(history))**2)
            
            if r_squared > self.config['pattern_detection_threshold']:
                trend_direction = 'increasing' if slope > 0 else 'decreasing'
                trends.append({
                    'type': 'trend',
                    'direction': trend_direction,
                    'strength': abs(slope),
                    'confidence': r_squared,
                    'domain': domain,
                    'world_line_id': world_line_id,
                    'time_range': (0, len(history) - 1)
                })
        
        return trends
    
    def _detect_cycle(self, history: List[float], domain: str, 
                     world_line_id: str) -> List[Dict[str, Any]]:
        """
        检测周期模式
        
        Args:
            history: 历史数据序列
            domain: 领域名称
            world_line_id: 世界线ID
            
        Returns:
            检测到的周期模式列表
        """
        cycles = []
        
        # 使用自相关检测周期性
        if len(history) >= 20:
            # 计算自相关
            n = len(history)
            mean = np.mean(history)
            variance = np.sum((history - mean)**2) / n
            
            if variance > 0:
                autocorr = []
                for lag in range(1, min(n//2, 30)):
                    c_k = np.sum((history[:-lag] - mean) * (history[lag:] - mean)) / n
                    autocorr.append(c_k / variance)
                
                # 寻找显著的峰值
                for i in range(1, len(autocorr)-1):
                    if (autocorr[i] > autocorr[i-1] and 
                        autocorr[i] > autocorr[i+1] and 
                        autocorr[i] > 0.5):
                        cycles.append({
                            'type': 'cycle',
                            'period': i + 1,
                            'strength': autocorr[i],
                            'confidence': min(autocorr[i], 1.0),
                            'domain': domain,
                            'world_line_id': world_line_id,
                            'time_range': (0, len(history) - 1)
                        })
        
        return cycles
    
    def _detect_sudden_change(self, history: List[float], domain: str, 
                             world_line_id: str) -> List[Dict[str, Any]]:
        """
        检测突变模式
        
        Args:
            history: 历史数据序列
            domain: 领域名称
            world_line_id: 世界线ID
            
        Returns:
            检测到的突变模式列表
        """
        changes = []
        
        # 检测相邻时间点的突变
        for i in range(1, len(history)):
            # 计算相对变化率
            if history[i-1] != 0:
                relative_change = abs(history[i] - history[i-1]) / abs(history[i-1])
            else:
                relative_change = abs(history[i] - history[i-1])
            
            # 检测突变
            if relative_change > 0.5:  # 50%以上的变化
                changes.append({
                    'type': 'sudden_change',
                    'magnitude': abs(history[i] - history[i-1]),
                    'relative_magnitude': relative_change,
                    'direction': 'increase' if history[i] > history[i-1] else 'decrease',
                    'domain': domain,
                    'world_line_id': world_line_id,
                    'time_point': i
                })
        
        return changes
    
    def _detect_convergence(self, history: List[float], domain: str, 
                           world_line_id: str) -> List[Dict[str, Any]]:
        """
        检测收敛模式
        
        Args:
            history: 历史数据序列
            domain: 领域名称
            world_line_id: 世界线ID
            
        Returns:
            检测到的收敛模式列表
        """
        convergences = []
        
        # 使用波动率减小检测收敛
        window_size = 5
        if len(history) >= window_size * 2:
            # 计算滑动窗口的标准差
            stds = []
            for i in range(len(history) - window_size + 1):
                window = history[i:i+window_size]
                stds.append(np.std(window))
            
            # 检测标准差是否持续减小
            if len(stds) >= 5:
                # 计算标准差序列的斜率
                x = np.arange(len(stds))
                slope = np.polyfit(x, stds, 1)[0]
                
                # 如果斜率显著为负，表示收敛
                if slope < -0.01 and np.std(history[-window_size:]) < 0.1:
                    convergences.append({
                        'type': 'convergence',
                        'stability': 1 / (abs(slope) + 0.01),
                        'target_value': np.mean(history[-window_size:]),
                        'confidence': min(abs(slope) * 10, 1.0),
                        'domain': domain,
                        'world_line_id': world_line_id,
                        'time_range': (len(history) - window_size, len(history) - 1)
                    })
        
        return convergences
    
    def _detect_divergence(self, history: List[float], domain: str, 
                          world_line_id: str) -> List[Dict[str, Any]]:
        """
        检测发散模式
        
        Args:
            history: 历史数据序列
            domain: 领域名称
            world_line_id: 世界线ID
            
        Returns:
            检测到的发散模式列表
        """
        divergences = []
        
        # 使用波动率增加检测发散
        window_size = 5
        if len(history) >= window_size * 2:
            # 计算滑动窗口的标准差
            stds = []
            for i in range(len(history) - window_size + 1):
                window = history[i:i+window_size]
                stds.append(np.std(window))
            
            # 检测标准差是否持续增大
            if len(stds) >= 5:
                # 计算标准差序列的斜率
                x = np.arange(len(stds))
                slope = np.polyfit(x, stds, 1)[0]
                
                # 如果斜率显著为正，表示发散
                if slope > 0.01:
                    divergences.append({
                        'type': 'divergence',
                        'volatility_increase': slope,
                        'current_volatility': np.std(history[-window_size:]),
                        'confidence': min(slope * 10, 1.0),
                        'domain': domain,
                        'world_line_id': world_line_id,
                        'time_range': (len(history) - window_size, len(history) - 1)
                    })
        
        return divergences
    
    def _merge_similar_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并相似的模式
        
        Args:
            patterns: 检测到的模式列表
            
        Returns:
            合并后的模式列表
        """
        # 按类型、领域分组
        grouped = defaultdict(list)
        for pattern in patterns:
            key = (pattern['type'], pattern['domain'])
            grouped[key].append(pattern)
        
        merged_patterns = []
        
        for key, group in grouped.items():
            if len(group) == 1:
                merged_patterns.extend(group)
                continue
            
            # 计算平均置信度
            avg_confidence = np.mean([p.get('confidence', 0) for p in group])
            
            # 如果平均置信度足够高，创建一个聚合模式
            if avg_confidence > self.config['pattern_detection_threshold']:
                merged = {
                    'type': key[0],
                    'domain': key[1],
                    'confidence': avg_confidence,
                    'occurrence_count': len(group),
                    'world_lines_affected': list(set(p['world_line_id'] for p in group))
                }
                
                # 添加特定类型的信息
                if key[0] == 'trend':
                    # 确定主要趋势方向
                    directions = [p['direction'] for p in group]
                    main_direction = max(set(directions), key=directions.count)
                    merged['direction'] = main_direction
                    merged['average_strength'] = np.mean(
                        [p['strength'] for p in group if p['direction'] == main_direction]
                    )
                elif key[0] == 'cycle':
                    # 计算平均周期
                    merged['average_period'] = np.mean([p['period'] for p in group])
                    merged['average_strength'] = np.mean([p['strength'] for p in group])
                
                merged_patterns.append(merged)
        
        return merged_patterns
    
    def _detect_turning_points(self, world_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        检测转折点
        
        Args:
            world_lines: 世界线列表
            
        Returns:
            检测到的转折点列表
        """
        turning_points = []
        sensitivity = self.config['turning_point_sensitivity']
        
        for world_line in world_lines:
            world_line_id = world_line.get('id', 'unknown')
            
            for domain in world_line.get('domains', {}):
                if 'history' not in world_line['domains'][domain]:
                    continue
                
                history = world_line['domains'][domain]['history']
                if len(history) < 5:
                    continue
                
                # 使用导数变化检测转折点
                for i in range(2, len(history) - 2):
                    # 计算相邻区间的斜率
                    slope1 = history[i] - history[i-2]
                    slope2 = history[i+2] - history[i]
                    
                    # 检查斜率符号变化
                    if slope1 * slope2 < 0:
                        # 计算变化幅度
                        change_magnitude = abs(slope1 - slope2) / 4
                        
                        if change_magnitude > sensitivity:
                            turning_points.append({
                                'domain': domain,
                                'world_line_id': world_line_id,
                                'time_point': i,
                                'magnitude': change_magnitude,
                                'direction_change': 'up_to_down' if slope1 > 0 else 'down_to_up',
                                'confidence': min(change_magnitude * 2, 1.0)
                            })
        
        return sorted(turning_points, key=lambda x: x['confidence'], reverse=True)
    
    def _extract_risk_signals(self, world_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        提取风险信号
        
        Args:
            world_lines: 世界线列表
            
        Returns:
            风险信号列表
        """
        risks = []
        risk_threshold = self.config['risk_threshold']
        
        for world_line in world_lines:
            world_line_id = world_line.get('id', 'unknown')
            
            # 检查各领域的风险指标
            for domain in world_line.get('domains', {}):
                domain_data = world_line['domains'][domain]
                
                # 检查风险指标
                if 'risk_score' in domain_data and domain_data['risk_score'] > risk_threshold:
                    risks.append({
                        'domain': domain,
                        'world_line_id': world_line_id,
                        'risk_score': domain_data['risk_score'],
                        'risk_level': 'high' if domain_data['risk_score'] > 0.9 else 'medium',
                        'risk_factors': domain_data.get('risk_factors', []),
                        'time_point': len(domain_data.get('history', [])) - 1
                    })
                
                # 检测异常值作为风险信号
                history = domain_data.get('history', [])
                if len(history) > 10:
                    mean = np.mean(history)
                    std = np.std(history)
                    
                    # 检查最近的数据是否为异常值
                    recent_value = history[-1]
                    if abs(recent_value - mean) > 2 * std:
                        risks.append({
                            'domain': domain,
                            'world_line_id': world_line_id,
                            'risk_score': min(1.0, (abs(recent_value - mean) / (3 * std))),
                            'risk_level': 'medium',
                            'risk_factors': ['anomaly_detected'],
                            'time_point': len(history) - 1,
                            'anomaly_magnitude': abs(recent_value - mean) / std
                        })
        
        # 按风险分数排序
        return sorted(risks, key=lambda x: x['risk_score'], reverse=True)
    
    def _identify_opportunity_windows(self, world_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        识别机会窗口
        
        Args:
            world_lines: 世界线列表
            
        Returns:
            机会窗口列表
        """
        opportunities = []
        opportunity_threshold = self.config['opportunity_threshold']
        
        for world_line in world_lines:
            world_line_id = world_line.get('id', 'unknown')
            
            # 检查各领域的机会指标
            for domain in world_line.get('domains', {}):
                domain_data = world_line['domains'][domain]
                
                # 检查机会指标
                if 'opportunity_score' in domain_data and domain_data['opportunity_score'] > opportunity_threshold:
                    opportunities.append({
                        'domain': domain,
                        'world_line_id': world_line_id,
                        'opportunity_score': domain_data['opportunity_score'],
                        'opportunity_type': domain_data.get('opportunity_type', 'general'),
                        'potential_impact': domain_data.get('potential_impact', 'medium'),
                        'time_point': len(domain_data.get('history', [])) - 1
                    })
                
                # 检测正向趋势作为机会信号
                history = domain_data.get('history', [])
                if len(history) > 5:
                    # 计算最近的趋势
                    x = np.arange(len(history[-5:]))
                    slope = np.polyfit(x, history[-5:], 1)[0]
                    
                    # 检查是否有强劲的正向趋势
                    if slope > 0.05:
                        opportunities.append({
                            'domain': domain,
                            'world_line_id': world_line_id,
                            'opportunity_score': min(1.0, slope * 20),
                            'opportunity_type': 'positive_trend',
                            'potential_impact': 'medium',
                            'time_point': len(history) - 1,
                            'trend_strength': slope
                        })
        
        # 按机会分数排序
        return sorted(opportunities, key=lambda x: x['opportunity_score'], reverse=True)
    
    def _calculate_confidence_scores(self, world_lines: List[Dict[str, Any]], 
                                   reality_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算洞察的置信度分数
        
        Args:
            world_lines: 世界线列表
            reality_data: 可选的现实数据
            
        Returns:
            各领域的置信度分数
        """
        confidence_scores = defaultdict(float)
        window_size = self.config['confidence_window']
        
        # 1. 计算世界线之间的一致性作为置信度基础
        domains = set()
        for world_line in world_lines:
            domains.update(world_line.get('domains', {}).keys())
        
        for domain in domains:
            # 收集所有世界线在该领域的最新值
            values = []
            for world_line in world_lines:
                if domain in world_line.get('domains', {}) and 'history' in world_line['domains'][domain]:
                    history = world_line['domains'][domain]['history']
                    if history:
                        values.append(history[-1])
            
            if len(values) > 1:
                # 使用一致性作为置信度指标（标准差越小，一致性越高）
                mean_value = np.mean(values)
                if mean_value != 0:
                    cv = np.std(values) / abs(mean_value)  # 变异系数
                    consistency_score = max(0, 1 - cv)  # 一致性分数
                else:
                    consistency_score = 1.0 if np.std(values) < 0.01 else 0.0
                
                confidence_scores[domain] = consistency_score
        
        # 2. 如果有现实数据，结合现实匹配度
        if reality_data:
            for domain in reality_data.get('domains', {}):
                if domain in confidence_scores:
                    reality_value = reality_data['domains'][domain].get('current_value')
                    if reality_value is not None:
                        # 计算世界线预测值与现实值的匹配度
                        predicted_values = []
                        for world_line in world_lines:
                            if domain in world_line.get('domains', {}) and 'history' in world_line['domains'][domain]:
                                history = world_line['domains'][domain]['history']
                                if history:
                                    predicted_values.append(history[-1])
                        
                        if predicted_values:
                            avg_prediction = np.mean(predicted_values)
                            if reality_value != 0:
                                prediction_error = abs(avg_prediction - reality_value) / abs(reality_value)
                                reality_match_score = max(0, 1 - prediction_error)
                            else:
                                reality_match_score = 1.0 if abs(avg_prediction) < 0.01 else 0.0
                            
                            # 结合一致性和现实匹配度
                            confidence_scores[domain] = 0.7 * confidence_scores[domain] + 0.3 * reality_match_score
        
        return dict(confidence_scores)
    
    def _summarize_key_metrics(self, world_lines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        汇总关键指标
        
        Args:
            world_lines: 世界线列表
            
        Returns:
            关键指标汇总
        """
        metrics = {
            'total_world_lines': len(world_lines),
            'domains_analyzed': set(),
            'average_confidence': {},
            'volatility_metrics': {},
            'correlation_metrics': {}
        }
        
        # 收集所有领域
        for world_line in world_lines:
            metrics['domains_analyzed'].update(world_line.get('domains', {}).keys())
        
        metrics['domains_analyzed'] = list(metrics['domains_analyzed'])
        
        # 计算每个领域的平均波动性
        for domain in metrics['domains_analyzed']:
            volatilities = []
            for world_line in world_lines:
                if domain in world_line.get('domains', {}) and 'history' in world_line['domains'][domain]:
                    history = world_line['domains'][domain]['history']
                    if len(history) > 1:
                        # 计算收益率的标准差作为波动性
                        returns = np.diff(history) / np.abs(history[:-1])
                        returns[np.isinf(returns)] = 0  # 处理无穷大
                        returns[np.isnan(returns)] = 0  # 处理NaN
                        if len(returns) > 0:
                            volatilities.append(np.std(returns))
            
            if volatilities:
                metrics['volatility_metrics'][domain] = {
                    'mean': np.mean(volatilities),
                    'std': np.std(volatilities),
                    'min': np.min(volatilities),
                    'max': np.max(volatilities)
                }
        
        return metrics


def create_default_insight_extractor() -> InsightExtractor:
    """
    创建默认的洞察提取器
    
    Returns:
        默认配置的InsightExtractor实例
    """
    return InsightExtractor()