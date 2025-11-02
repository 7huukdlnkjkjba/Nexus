#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
领域交互模型 (Interaction Model)
负责处理经济、政治、技术、气候等领域间的相互影响
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class InteractionModel:
    """
    领域交互模型类，管理各领域之间的相互影响关系
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化领域交互模型
        
        Args:
            config: 配置字典，包含交互模型的参数
        """
        # 默认配置
        self.config = {
            'domains': ['economic', 'political', 'technological', 'climate'],
            'default_influence_strength': 0.1,
            'nonlinear_factor': 0.3,
            'cascade_threshold': 0.5,
            'max_cascade_depth': 5
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        self.domains = self.config['domains']
        self.num_domains = len(self.domains)
        self.domain_index = {domain: i for i, domain in enumerate(self.domains)}
        
        # 初始化影响矩阵
        self.influence_matrix = self._initialize_influence_matrix()
        
        # 初始化非线性反馈系数
        self.feedback_coefficients = self._initialize_feedback_coefficients()
        
        logger.info(f"InteractionModel initialized with {self.num_domains} domains")
    
    def _initialize_influence_matrix(self) -> np.ndarray:
        """
        初始化领域间影响矩阵
        
        Returns:
            numpy数组，表示领域间的影响强度
        """
        # 创建默认影响矩阵
        matrix = np.ones((self.num_domains, self.num_domains)) * self.config['default_influence_strength']
        
        # 设置自影响为1.0（完全影响自身）
        np.fill_diagonal(matrix, 1.0)
        
        # 根据领域特性调整影响强度
        self._adjust_domain_influences(matrix)
        
        return matrix
    
    def _adjust_domain_influences(self, matrix: np.ndarray) -> None:
        """
        根据领域特性调整影响矩阵
        
        Args:
            matrix: 影响矩阵
        """
        # 经济对政治的影响较强
        if 'economic' in self.domain_index and 'political' in self.domain_index:
            matrix[self.domain_index['economic'], self.domain_index['political']] = 0.3
        
        # 政治对经济的影响较强
        if 'political' in self.domain_index and 'economic' in self.domain_index:
            matrix[self.domain_index['political'], self.domain_index['economic']] = 0.35
        
        # 技术对经济的影响较强
        if 'technological' in self.domain_index and 'economic' in self.domain_index:
            matrix[self.domain_index['technological'], self.domain_index['economic']] = 0.25
        
        # 气候对经济的影响中等
        if 'climate' in self.domain_index and 'economic' in self.domain_index:
            matrix[self.domain_index['climate'], self.domain_index['economic']] = 0.2
        
        # 经济对气候的影响中等
        if 'economic' in self.domain_index and 'climate' in self.domain_index:
            matrix[self.domain_index['economic'], self.domain_index['climate']] = 0.15
    
    def _initialize_feedback_coefficients(self) -> Dict[str, float]:
        """
        初始化非线性反馈系数
        
        Returns:
            各领域的反馈系数字典
        """
        return {
            'economic': 0.4,  # 经济系统有较强的自我调节能力
            'political': 0.6,  # 政治系统反馈较强烈
            'technological': 0.7,  # 技术发展有较强的正反馈
            'climate': 0.2   # 气候系统反馈较慢
        }
    
    def calculate_interactions(self, domain_states: Dict[str, float], 
                              domain_changes: Dict[str, float]) -> Dict[str, float]:
        """
        计算领域间的相互影响
        
        Args:
            domain_states: 各领域当前状态值
            domain_changes: 各领域自身变化值
            
        Returns:
            各领域受其他领域影响后的最终变化值
        """
        # 转换为向量形式
        state_vector = np.array([domain_states.get(domain, 0.0) for domain in self.domains])
        change_vector = np.array([domain_changes.get(domain, 0.0) for domain in self.domains])
        
        # 计算基础影响
        base_influences = np.dot(self.influence_matrix, change_vector)
        
        # 应用非线性反馈
        final_changes = self._apply_nonlinear_feedback(state_vector, base_influences)
        
        # 检测并处理级联效应
        cascade_effects = self._detect_cascade_effects(final_changes)
        if cascade_effects.any():
            final_changes = self._process_cascade_effects(
                state_vector, final_changes, cascade_effects
            )
        
        # 转换回字典形式
        result = {domain: final_changes[i] for i, domain in enumerate(self.domains)}
        
        return result
    
    def _apply_nonlinear_feedback(self, state_vector: np.ndarray, 
                                 influences: np.ndarray) -> np.ndarray:
        """
        应用非线性反馈机制
        
        Args:
            state_vector: 领域状态向量
            influences: 基础影响向量
            
        Returns:
            应用非线性反馈后的影响向量
        """
        feedback_strength = self.config['nonlinear_factor']
        
        # 计算每个领域的非线性反馈
        final_influences = np.zeros_like(influences)
        
        for i, domain in enumerate(self.domains):
            # 获取该领域的反馈系数
            feedback_coef = self.feedback_coefficients.get(domain, 0.5)
            
            # 非线性反馈公式: 影响 * (1 + 反馈强度 * 状态 * 反馈系数)
            # 当状态值较高时，正反馈会放大影响，负反馈会减弱影响
            feedback_effect = 1.0 + feedback_strength * state_vector[i] * feedback_coef
            final_influences[i] = influences[i] * feedback_effect
        
        return final_influences
    
    def _detect_cascade_effects(self, changes: np.ndarray) -> np.ndarray:
        """
        检测级联效应触发点
        
        Args:
            changes: 领域变化向量
            
        Returns:
            布尔向量，表示哪些领域触发了级联效应
        """
        threshold = self.config['cascade_threshold']
        return np.abs(changes) > threshold
    
    def _process_cascade_effects(self, state_vector: np.ndarray, 
                               changes: np.ndarray, 
                               cascade_triggers: np.ndarray) -> np.ndarray:
        """
        处理级联效应
        
        Args:
            state_vector: 领域状态向量
            changes: 初始变化向量
            cascade_triggers: 级联效应触发点
            
        Returns:
            处理级联效应后的最终变化向量
        """
        max_depth = self.config['max_cascade_depth']
        current_changes = changes.copy()
        
        for depth in range(max_depth):
            # 计算当前变化引起的次级影响
            secondary_influences = np.dot(self.influence_matrix, current_changes) * 0.5  # 次级影响减半
            
            # 只在触发点应用次级影响
            new_changes = current_changes.copy()
            new_changes[cascade_triggers] += secondary_influences[cascade_triggers]
            
            # 检查变化是否收敛
            if np.max(np.abs(new_changes - current_changes)) < 0.01:
                break
            
            current_changes = new_changes
        
        logger.debug(f"Processed cascade effects up to depth {depth+1}")
        return current_changes
    
    def update_influence_matrix(self, new_relationships: Dict[Tuple[str, str], float]) -> None:
        """
        更新领域间影响关系
        
        Args:
            new_relationships: 新的影响关系，键为(源领域, 目标领域)，值为影响强度
        """
        for (source, target), strength in new_relationships.items():
            if source in self.domain_index and target in self.domain_index:
                i, j = self.domain_index[source], self.domain_index[target]
                self.influence_matrix[i, j] = strength
                logger.debug(f"Updated influence from {source} to {target}: {strength}")
    
    def get_influence_network(self) -> Dict[str, Dict[str, float]]:
        """
        获取影响网络的可读表示
        
        Returns:
            嵌套字典，表示领域间的影响关系
        """
        network = {}
        for source in self.domains:
            network[source] = {}
            for target in self.domains:
                if source != target:
                    i, j = self.domain_index[source], self.domain_index[target]
                    network[source][target] = self.influence_matrix[i, j]
        
        return network
    
    def validate_model(self) -> bool:
        """
        验证模型参数的有效性
        
        Returns:
            模型是否有效
        """
        # 检查影响矩阵是否为有效矩阵
        if not isinstance(self.influence_matrix, np.ndarray) or \
           self.influence_matrix.shape != (self.num_domains, self.num_domains):
            logger.error("Invalid influence matrix")
            return False
        
        # 检查参数范围
        if not (0 <= self.config['nonlinear_factor'] <= 1):
            logger.error("Nonlinear factor must be between 0 and 1")
            return False
        
        if not (0 <= self.config['cascade_threshold'] <= 1):
            logger.error("Cascade threshold must be between 0 and 1")
            return False
        
        if self.config['max_cascade_depth'] < 1:
            logger.error("Max cascade depth must be at least 1")
            return False
        
        return True


def create_default_interaction_model() -> InteractionModel:
    """
    创建默认的领域交互模型
    
    Returns:
        默认配置的InteractionModel实例
    """
    return InteractionModel()