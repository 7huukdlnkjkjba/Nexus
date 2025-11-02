#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
随机数工具模块 (Random Utils)
提供带种子的随机数生成器，确保模拟的可重现性
"""

import random
import numpy as np
import hashlib
import time
from typing import Any, List, Tuple, Optional, Dict


class SeedGenerator:
    """
    种子生成器，用于创建确定性的随机种子
    """
    
    @staticmethod
    def from_string(seed_str: str) -> int:
        """
        从字符串生成整数种子
        
        Args:
            seed_str: 输入字符串
            
        Returns:
            整数种子
        """
        # 使用MD5哈希生成种子
        hash_obj = hashlib.md5(seed_str.encode('utf-8'))
        # 取哈希值的前8位作为种子
        return int(hash_obj.hexdigest()[:8], 16)
    
    @staticmethod
    def from_time() -> int:
        """
        从当前时间生成种子
        
        Returns:
            整数种子
        """
        return int(time.time() * 1000) % (2**32 - 1)
    
    @staticmethod
    def from_object(obj: Any) -> int:
        """
        从任意对象生成种子
        
        Args:
            obj: 任意可序列化对象
            
        Returns:
            整数种子
        """
        obj_str = str(obj)
        return SeedGenerator.from_string(obj_str)
    
    @staticmethod
    def combine_seeds(seeds: List[int]) -> int:
        """
        组合多个种子为一个
        
        Args:
            seeds: 种子列表
            
        Returns:
            组合后的种子
        """
        # 使用异或操作组合种子
        combined = 0
        for seed in seeds:
            combined ^= seed
        return combined


class RandomGenerator:
    """
    随机数生成器，封装Python和NumPy的随机数功能
    确保使用相同种子时产生相同的随机序列
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化随机数生成器
        
        Args:
            seed: 随机种子，如果为None则使用当前时间
        """
        self.seed = seed if seed is not None else SeedGenerator.from_time()
        self._python_rng = random.Random(self.seed)
        self._numpy_rng = np.random.RandomState(self.seed)
        
        # 记录初始种子
        self.initial_seed = self.seed
    
    def reset(self) -> None:
        """
        重置随机数生成器到初始种子
        """
        self.seed = self.initial_seed
        self._python_rng = random.Random(self.seed)
        self._numpy_rng = np.random.RandomState(self.seed)
    
    def set_seed(self, seed: int) -> None:
        """
        设置新的随机种子
        
        Args:
            seed: 新的随机种子
        """
        self.seed = seed
        self._python_rng = random.Random(self.seed)
        self._numpy_rng = np.random.RandomState(self.seed)
    
    # Python random 模块方法
    def random(self) -> float:
        """
        生成[0.0, 1.0)之间的随机浮点数
        
        Returns:
            随机浮点数
        """
        return self._python_rng.random()
    
    def randint(self, a: int, b: int) -> int:
        """
        生成[a, b]之间的随机整数
        
        Args:
            a: 最小值（包含）
            b: 最大值（包含）
            
        Returns:
            随机整数
        """
        return self._python_rng.randint(a, b)
    
    def choice(self, seq: List[Any]) -> Any:
        """
        从序列中随机选择一个元素
        
        Args:
            seq: 序列
            
        Returns:
            随机选择的元素
        """
        return self._python_rng.choice(seq)
    
    def choices(self, population: List[Any], weights: Optional[List[float]] = None,
                *, cum_weights: Optional[List[float]] = None, k: int = 1) -> List[Any]:
        """
        从序列中随机选择多个元素（有放回）
        
        Args:
            population: 序列
            weights: 权重序列
            cum_weights: 累积权重序列
            k: 选择数量
            
        Returns:
            选择的元素列表
        """
        return self._python_rng.choices(population, weights=weights,
                                      cum_weights=cum_weights, k=k)
    
    def sample(self, population: List[Any], k: int) -> List[Any]:
        """
        从序列中随机选择多个元素（无放回）
        
        Args:
            population: 序列
            k: 选择数量
            
        Returns:
            选择的元素列表
        """
        return self._python_rng.sample(population, k)
    
    def shuffle(self, x: List[Any]) -> None:
        """
        随机打乱序列
        
        Args:
            x: 要打乱的序列
        """
        self._python_rng.shuffle(x)
    
    def uniform(self, a: float, b: float) -> float:
        """
        生成[a, b)之间的均匀分布随机浮点数
        
        Args:
            a: 最小值
            b: 最大值
            
        Returns:
            随机浮点数
        """
        return self._python_rng.uniform(a, b)
    
    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """
        生成高斯分布的随机浮点数
        
        Args:
            mu: 均值
            sigma: 标准差
            
        Returns:
            随机浮点数
        """
        return self._python_rng.gauss(mu, sigma)
    
    # NumPy random 模块方法
    def np_random(self, size: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """
        生成[0.0, 1.0)之间的随机浮点数数组
        
        Args:
            size: 输出形状
            
        Returns:
            随机浮点数数组
        """
        return self._numpy_rng.random(size=size)
    
    def np_randint(self, low: int, high: Optional[int] = None, 
                   size: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """
        生成随机整数数组
        
        Args:
            low: 最小值
            high: 最大值（不包含）
            size: 输出形状
            
        Returns:
            随机整数数组
        """
        return self._numpy_rng.randint(low, high=high, size=size)
    
    def np_normal(self, loc: float = 0.0, scale: float = 1.0, 
                 size: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """
        生成正态分布的随机数数组
        
        Args:
            loc: 均值
            scale: 标准差
            size: 输出形状
            
        Returns:
            随机数数组
        """
        return self._numpy_rng.normal(loc=loc, scale=scale, size=size)
    
    def np_uniform(self, low: float = 0.0, high: float = 1.0,
                  size: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """
        生成均匀分布的随机数数组
        
        Args:
            low: 最小值
            high: 最大值
            size: 输出形状
            
        Returns:
            随机数数组
        """
        return self._numpy_rng.uniform(low=low, high=high, size=size)
    
    def np_shuffle(self, x: np.ndarray) -> None:
        """
        随机打乱数组
        
        Args:
            x: 要打乱的数组
        """
        self._numpy_rng.shuffle(x)
    
    def np_choice(self, a: Any, size: Optional[Tuple[int, ...]] = None,
                 replace: bool = True, p: Optional[np.ndarray] = None) -> np.ndarray:
        """
        从数组中随机选择元素
        
        Args:
            a: 一维数组或整数
            size: 输出形状
            replace: 是否有放回
            p: 概率权重数组
            
        Returns:
            选择的元素数组
        """
        return self._numpy_rng.choice(a, size=size, replace=replace, p=p)


class RandomContext:
    """
    随机数上下文管理器，用于临时保存和恢复随机数状态
    """
    
    def __init__(self, rng: RandomGenerator):
        self.rng = rng
        self.old_python_state = None
        self.old_numpy_state = None
    
    def __enter__(self):
        # 保存当前状态
        self.old_python_state = self.rng._python_rng.getstate()
        self.old_numpy_state = self.rng._numpy_rng.get_state()
        return self.rng
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复之前的状态
        self.rng._python_rng.setstate(self.old_python_state)
        self.rng._numpy_rng.set_state(self.old_numpy_state)


class DeterministicRandom:
    """
    确定性随机数生成器的工厂类，提供全局访问点
    """
    
    _rng_instances: Dict[str, RandomGenerator] = {}
    
    @classmethod
    def get_instance(cls, name: str = 'default', 
                    seed: Optional[int] = None) -> RandomGenerator:
        """
        获取指定名称的随机数生成器实例
        
        Args:
            name: 实例名称
            seed: 随机种子
            
        Returns:
            随机数生成器实例
        """
        if name not in cls._rng_instances:
            cls._rng_instances[name] = RandomGenerator(seed)
        elif seed is not None and cls._rng_instances[name].seed != seed:
            # 如果提供了新种子，则重新初始化
            cls._rng_instances[name] = RandomGenerator(seed)
        
        return cls._rng_instances[name]
    
    @classmethod
    def reset_all(cls) -> None:
        """
        重置所有随机数生成器实例
        """
        for rng in cls._rng_instances.values():
            rng.reset()
    
    @classmethod
    def clear_all(cls) -> None:
        """
        清除所有随机数生成器实例
        """
        cls._rng_instances.clear()


# 随机扰动函数
def apply_random_perturbation(value: float, magnitude: float = 0.01,
                             rng: Optional[RandomGenerator] = None) -> float:
    """
    对数值应用随机扰动
    
    Args:
        value: 原始值
        magnitude: 扰动幅度（相对值）
        rng: 随机数生成器
        
    Returns:
        扰动后的值
    """
    if rng is None:
        rng = DeterministicRandom.get_instance()
    
    # 生成[-magnitude, magnitude]之间的随机扰动
    perturbation = rng.uniform(-magnitude, magnitude)
    return value * (1 + perturbation)


def generate_correlated_randoms(count: int, correlation: float = 0.5,
                               mean: float = 0.0, std: float = 1.0,
                               rng: Optional[RandomGenerator] = None) -> List[float]:
    """
    生成具有指定相关性的随机数序列
    
    Args:
        count: 随机数数量
        correlation: 自相关系数
        mean: 均值
        std: 标准差
        rng: 随机数生成器
        
    Returns:
        相关随机数列表
    """
    if rng is None:
        rng = DeterministicRandom.get_instance()
    
    # 使用AR(1)模型生成相关序列
    result = [rng.normal(mean, std)]
    
    for i in range(1, count):
        # x_t = correlation * x_{t-1} + e_t
        noise = rng.normal(0, std * np.sqrt(1 - correlation**2))
        next_value = correlation * result[-1] + noise
        result.append(next_value)
    
    # 调整均值和标准差
    result_array = np.array(result)
    result_array = (result_array - np.mean(result_array)) / np.std(result_array)
    result_array = result_array * std + mean
    
    return result_array.tolist()


def generate_scenario_weights(scenarios: List[str], base_weights: Optional[Dict[str, float]] = None,
                             uncertainty: float = 0.1,
                             rng: Optional[RandomGenerator] = None) -> Dict[str, float]:
    """
    生成场景权重，在基础权重上添加随机波动
    
    Args:
        scenarios: 场景列表
        base_weights: 基础权重字典
        uncertainty: 不确定性水平
        rng: 随机数生成器
        
    Returns:
        调整后的权重字典
    """
    if rng is None:
        rng = DeterministicRandom.get_instance()
    
    # 默认等权重
    if base_weights is None:
        base_weights = {scenario: 1.0 / len(scenarios) for scenario in scenarios}
    
    # 应用随机扰动
    weights = {}
    total_weight = 0.0
    
    for scenario in scenarios:
        base = base_weights.get(scenario, 1.0 / len(scenarios))
        perturbed = max(0.0, apply_random_perturbation(base, uncertainty, rng))
        weights[scenario] = perturbed
        total_weight += perturbed
    
    # 归一化
    if total_weight > 0:
        for scenario in weights:
            weights[scenario] /= total_weight
    
    return weights


# 初始化默认随机数生成器
default_rng = DeterministicRandom.get_instance('default')

# 提供便捷的随机函数
def get_random() -> float:
    return default_rng.random()

def get_randint(a: int, b: int) -> int:
    return default_rng.randint(a, b)

def get_choice(seq: List[Any]) -> Any:
    return default_rng.choice(seq)

def get_uniform(a: float, b: float) -> float:
    return default_rng.uniform(a, b)

def get_normal(mu: float = 0.0, sigma: float = 1.0) -> float:
    return default_rng.gauss(mu, sigma)