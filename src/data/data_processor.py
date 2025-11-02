#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理模块 (Data Processor)
负责数据清洗、标准化、特征工程等操作
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import statsmodels.api as sm

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.data_types import DomainType, RealityData, WorldLine

logger = get_logger(__name__)


class DataProcessor:
    """
    数据处理器基类
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据处理器
        
        Args:
            config: 配置信息
        """
        self.config = config or {}
        self.logger = logger
    
    def process(self, data: Any, **kwargs) -> Any:
        """
        处理数据的主方法，需要子类实现
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            处理后的数据
        """
        raise NotImplementedError("Subclasses must implement process method")


class RealityDataProcessor(DataProcessor):
    """
    现实数据处理器
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化现实数据处理器
        
        Args:
            config: 配置信息
        """
        default_config = {
            'imputation_strategy': 'mean',  # mean, median, most_frequent, knn
            'knn_neighbors': 5,
            'outlier_detection': 'zscore',  # zscore, iqr, none
            'zscore_threshold': 3.0,
            'iqr_multiplier': 1.5,
            'normalize_method': 'minmax',  # minmax, standard, robust, none
            'window_size': 5,  # 用于计算移动平均线等
            'max_gap_years': 3  # 最大允许的年份间隔
        }
        merged_config = {**default_config, **(config or {})}
        super().__init__(merged_config)
        
        # 初始化转换器
        self.scalers: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
    
    def process(self, reality_data: RealityData, **kwargs) -> RealityData:
        """
        处理现实数据
        
        Args:
            reality_data: 原始现实数据
            **kwargs: 处理参数
            
        Returns:
            处理后的现实数据
        """
        logger.info(f"Processing reality data from {reality_data.source}")
        
        # 创建处理后的数据副本
        processed_data = RealityData(
            name=f"processed_{reality_data.name}",
            source=reality_data.source,
            metadata={**reality_data.metadata, 'processed': True}
        )
        
        # 处理每个指标的数据
        for (domain_type, metric_name), data_points in reality_data.data_points.items():
            try:
                # 转换为DataFrame便于处理
                df = pd.DataFrame(data_points, columns=['year', 'value'])
                df = df.sort_values('year')
                
                # 数据清洗
                cleaned_df = self._clean_data(df, domain_type, metric_name)
                
                # 处理缺失值
                imputed_df = self._impute_missing_values(cleaned_df, domain_type, metric_name)
                
                # 异常值检测和处理
                outlier_handled_df = self._handle_outliers(imputed_df, domain_type, metric_name)
                
                # 数据标准化/归一化
                normalized_df = self._normalize_data(outlier_handled_df, domain_type, metric_name)
                
                # 特征工程
                features_df = self._feature_engineering(normalized_df, domain_type, metric_name)
                
                # 将处理后的数据添加到结果中
                for _, row in features_df.iterrows():
                    processed_data.add_data_point(domain_type, metric_name, int(row['year']), float(row['value']))
                    
                    # 添加生成的特征
                    for col in features_df.columns:
                        if col not in ['year', 'value']:
                            feature_name = f"{metric_name}_{col}"
                            processed_data.add_data_point(domain_type, feature_name, int(row['year']), float(row[col]))
                
                logger.debug(f"Processed {metric_name}: {len(features_df)} data points")
                
            except Exception as e:
                logger.error(f"Error processing {metric_name}: {e}")
                # 如果处理失败，保留原始数据
                for year, value in data_points:
                    processed_data.add_data_point(domain_type, metric_name, year, value)
        
        return processed_data
    
    def _clean_data(self, df: pd.DataFrame, domain_type: DomainType,
                   metric_name: str) -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            df: 输入DataFrame
            domain_type: 领域类型
            metric_name: 指标名称
            
        Returns:
            清洗后的DataFrame
        """
        # 复制数据以避免修改原始数据
        cleaned_df = df.copy()
        
        # 删除重复行
        cleaned_df = cleaned_df.drop_duplicates(subset=['year'])
        
        # 删除明显无效的值
        cleaned_df = cleaned_df[pd.to_numeric(cleaned_df['value'], errors='coerce').notna()]
        
        # 检查时间序列的连续性
        years = cleaned_df['year'].sort_values().tolist()
        max_gap = self.config['max_gap_years']
        
        if len(years) > 1:
            gaps = []
            for i in range(1, len(years)):
                gap = years[i] - years[i-1]
                if gap > max_gap:
                    gaps.append((years[i-1], years[i], gap))
            
            if gaps:
                logger.warning(f"Large gaps detected in {metric_name}: {gaps}")
        
        # 基于领域的特定清洗逻辑
        if domain_type == DomainType.ECONOMIC:
            # 经济指标通常不能为负数（除了增长率等）
            if 'growth' not in metric_name.lower() and 'rate' not in metric_name.lower():
                cleaned_df = cleaned_df[cleaned_df['value'] >= 0]
        
        return cleaned_df
    
    def _impute_missing_values(self, df: pd.DataFrame, domain_type: DomainType,
                              metric_name: str) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 输入DataFrame
            domain_type: 领域类型
            metric_name: 指标名称
            
        Returns:
            处理缺失值后的DataFrame
        """
        imputed_df = df.copy()
        strategy = self.config['imputation_strategy']
        
        # 生成完整的年份序列
        min_year = df['year'].min()
        max_year = df['year'].max()
        full_years = pd.DataFrame({'year': range(min_year, max_year + 1)})
        
        # 合并以识别缺失的年份
        merged_df = pd.merge(full_years, imputed_df, on='year', how='left')
        
        # 获取需要插值的数据
        values = merged_df['value'].values.reshape(-1, 1)
        
        if strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
            merged_df['value'] = imputer.fit_transform(values).flatten()
        elif strategy == 'median':
            imputer = SimpleImputer(strategy='median')
            merged_df['value'] = imputer.fit_transform(values).flatten()
        elif strategy == 'most_frequent':
            imputer = SimpleImputer(strategy='most_frequent')
            merged_df['value'] = imputer.fit_transform(values).flatten()
        elif strategy == 'knn':
            # 为KNN插值创建年份特征
            year_features = merged_df['year'].values.reshape(-1, 1)
            # 只对非NaN值进行训练
            valid_mask = ~np.isnan(values.flatten())
            
            if valid_mask.sum() >= self.config['knn_neighbors']:
                # 使用KNN插值
                from sklearn.neighbors import KNeighborsRegressor
                knn = KNeighborsRegressor(n_neighbors=self.config['knn_neighbors'])
                knn.fit(year_features[valid_mask], values[valid_mask])
                merged_df.loc[~valid_mask, 'value'] = knn.predict(year_features[~valid_mask]).flatten()
            else:
                # 如果有效数据点太少，回退到线性插值
                merged_df['value'] = merged_df['value'].interpolate(method='linear')
        else:
            # 默认使用线性插值
            merged_df['value'] = merged_df['value'].interpolate(method='linear')
        
        # 保存imputer供后续使用
        imputer_key = f"{domain_type.name}_{metric_name}"
        self.imputers[imputer_key] = imputer
        
        return merged_df
    
    def _handle_outliers(self, df: pd.DataFrame, domain_type: DomainType,
                        metric_name: str) -> pd.DataFrame:
        """
        异常值检测和处理
        
        Args:
            df: 输入DataFrame
            domain_type: 领域类型
            metric_name: 指标名称
            
        Returns:
            处理异常值后的DataFrame
        """
        outlier_df = df.copy()
        method = self.config['outlier_detection']
        
        if method == 'zscore':
            # Z-Score方法
            z_scores = np.abs(stats.zscore(df['value']))
            threshold = self.config['zscore_threshold']
            outliers = z_scores > threshold
            
            if outliers.any():
                # 用中位数替换异常值
                median_val = df['value'].median()
                outlier_df.loc[outliers, 'value'] = median_val
                logger.debug(f"Replaced {outliers.sum()} outliers in {metric_name} using zscore method")
        
        elif method == 'iqr':
            # IQR方法
            Q1 = df['value'].quantile(0.25)
            Q3 = df['value'].quantile(0.75)
            IQR = Q3 - Q1
            multiplier = self.config['iqr_multiplier']
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers = (df['value'] < lower_bound) | (df['value'] > upper_bound)
            
            if outliers.any():
                # 用边界值替换异常值
                outlier_df.loc[df['value'] < lower_bound, 'value'] = lower_bound
                outlier_df.loc[df['value'] > upper_bound, 'value'] = upper_bound
                logger.debug(f"Replaced {outliers.sum()} outliers in {metric_name} using IQR method")
        
        return outlier_df
    
    def _normalize_data(self, df: pd.DataFrame, domain_type: DomainType,
                       metric_name: str) -> pd.DataFrame:
        """
        数据标准化/归一化
        
        Args:
            df: 输入DataFrame
            domain_type: 领域类型
            metric_name: 指标名称
            
        Returns:
            标准化后的DataFrame
        """
        normalized_df = df.copy()
        method = self.config['normalize_method']
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            # 不进行标准化
            return normalized_df
        
        # 应用标准化
        values = normalized_df['value'].values.reshape(-1, 1)
        normalized_df['value'] = scaler.fit_transform(values).flatten()
        
        # 保存scaler供后续使用
        scaler_key = f"{domain_type.name}_{metric_name}"
        self.scalers[scaler_key] = scaler
        
        return normalized_df
    
    def _feature_engineering(self, df: pd.DataFrame, domain_type: DomainType,
                           metric_name: str) -> pd.DataFrame:
        """
        特征工程
        
        Args:
            df: 输入DataFrame
            domain_type: 领域类型
            metric_name: 指标名称
            
        Returns:
            包含新特征的DataFrame
        """
        features_df = df.copy()
        window_size = self.config['window_size']
        
        # 计算移动平均值
        features_df['rolling_mean'] = features_df['value'].rolling(window=window_size, min_periods=1).mean()
        
        # 计算移动标准差
        features_df['rolling_std'] = features_df['value'].rolling(window=window_size, min_periods=1).std().fillna(0)
        
        # 计算变化率（同比）
        features_df['yearly_change'] = features_df['value'].pct_change(1).fillna(0)
        
        # 计算趋势（简单线性回归斜率）
        if len(features_df) >= 2:
            x = features_df['year'].values.reshape(-1, 1)
            y = features_df['value'].values
            
            # 简单线性回归
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(x, y)
            
            # 预测值作为趋势
            features_df['trend'] = model.predict(x)
            
            # 趋势斜率
            features_df['trend_slope'] = model.coef_[0]
        
        # 计算季节性（如果数据足够长）
        if len(features_df) >= 10:
            # 使用移动窗口的季节性分解（简化版）
            # 计算相对于移动平均的偏差
            features_df['seasonal_deviation'] = features_df['value'] - features_df['rolling_mean']
        
        # 计算波动率
        features_df['volatility'] = features_df['rolling_std'] / features_df['rolling_mean'].replace(0, np.nan).fillna(0)
        
        # 计算加速/减速
        features_df['acceleration'] = features_df['yearly_change'].diff().fillna(0)
        
        # 领域特定的特征工程
        if domain_type == DomainType.ECONOMIC:
            # 经济指标可能有周期性特征
            if 'gdp' in metric_name.lower():
                # 计算GDP增长率的波动率
                pass
        
        elif domain_type == DomainType.CLIMATE:
            # 气候数据的额外特征
            if 'temperature' in metric_name.lower():
                # 计算5年平均变化率
                features_df['5yr_temp_change'] = features_df['value'].pct_change(5).fillna(0)
        
        return features_df
    
    def denormalize(self, value: float, domain_type: DomainType,
                   metric_name: str) -> float:
        """
        反标准化数据
        
        Args:
            value: 标准化后的值
            domain_type: 领域类型
            metric_name: 指标名称
            
        Returns:
            原始范围的值
        """
        scaler_key = f"{domain_type.name}_{metric_name}"
        if scaler_key in self.scalers:
            scaler = self.scalers[scaler_key]
            return scaler.inverse_transform([[value]])[0, 0]
        return value


class WorldLineProcessor(DataProcessor):
    """
    世界线数据处理器
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化世界线数据处理器
        
        Args:
            config: 配置信息
        """
        default_config = {
            'normalize_metrics': True,
            'calculate_trends': True,
            'detect_volatility': True,
            'tipping_point_threshold': 0.9,
            'correlation_window': 10
        }
        merged_config = {**default_config, **(config or {})}
        super().__init__(merged_config)
        self.reality_processor = RealityDataProcessor()
    
    def process(self, worldline: WorldLine, **kwargs) -> WorldLine:
        """
        处理世界线数据
        
        Args:
            worldline: 原始世界线
            **kwargs: 处理参数
            
        Returns:
            处理后的世界线
        """
        logger.info(f"Processing worldline: {worldline.world_id}")
        
        # 处理每个领域的状态
        for domain_type, domain_state in worldline.domain_states.items():
            try:
                # 标准化指标
                if self.config['normalize_metrics']:
                    self._normalize_domain_metrics(domain_state)
                
                # 计算趋势
                if self.config['calculate_trends']:
                    self._calculate_trends(worldline, domain_type, domain_state)
                
                # 检测波动率
                if self.config['detect_volatility']:
                    self._calculate_volatility(worldline, domain_type, domain_state)
                
                # 检测临界点距离
                self._detect_tipping_points(domain_state)
                
                # 分析依赖关系
                self._analyze_dependencies(worldline, domain_type, domain_state)
                
            except Exception as e:
                logger.error(f"Error processing {domain_type} in worldline {worldline.world_id}: {e}")
        
        return worldline
    
    def _normalize_domain_metrics(self, domain_state: Any) -> None:
        """
        标准化领域指标
        
        Args:
            domain_state: 领域状态
        """
        if not domain_state.metrics:
            return
        
        values = np.array(list(domain_state.metrics.values())).reshape(-1, 1)
        if len(values) > 0:
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(values).flatten()
            
            # 更新标准化后的指标
            for i, (metric_name, _) in enumerate(domain_state.metrics.items()):
                # 保存原始值在趋势字段中（临时使用）
                if metric_name not in domain_state.trends:
                    domain_state.trends[metric_name] = domain_state.metrics[metric_name]
                # 更新为标准化后的值
                domain_state.metrics[metric_name] = scaled_values[i]
    
    def _calculate_trends(self, worldline: WorldLine, domain_type: DomainType,
                         domain_state: Any) -> None:
        """
        计算趋势
        
        Args:
            worldline: 世界线
            domain_type: 领域类型
            domain_state: 领域状态
        """
        # 从历史数据计算趋势
        if len(worldline.history) >= 2:
            for metric_name in domain_state.metrics:
                # 收集历史值
                historical_values = []
                historical_years = []
                
                # 从历史记录中获取
                for history_entry in worldline.history:
                    if domain_type in history_entry['states']:
                        hist_state = history_entry['states'][domain_type]
                        if metric_name in hist_state.metrics:
                            historical_values.append(hist_state.metrics[metric_name])
                            historical_years.append(history_entry['year'])
                
                # 添加当前值
                historical_values.append(domain_state.metrics[metric_name])
                historical_years.append(worldline.current_year)
                
                if len(historical_values) >= 2:
                    # 计算简单趋势（线性回归斜率）
                    x = np.array(historical_years).reshape(-1, 1)
                    y = np.array(historical_values)
                    
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model.fit(x, y)
                    
                    # 保存趋势
                    domain_state.trends[metric_name] = float(model.coef_[0])
    
    def _calculate_volatility(self, worldline: WorldLine, domain_type: DomainType,
                            domain_state: Any) -> None:
        """
        计算波动率
        
        Args:
            worldline: 世界线
            domain_type: 领域类型
            domain_state: 领域状态
        """
        # 从历史数据计算波动率
        if len(worldline.history) >= 3:
            for metric_name in domain_state.metrics:
                # 收集历史值
                historical_values = []
                
                # 从历史记录中获取
                for history_entry in worldline.history[-5:]:  # 只看最近5个时间点
                    if domain_type in history_entry['states']:
                        hist_state = history_entry['states'][domain_type]
                        if metric_name in hist_state.metrics:
                            historical_values.append(hist_state.metrics[metric_name])
                
                # 添加当前值
                historical_values.append(domain_state.metrics[metric_name])
                
                if len(historical_values) >= 3:
                    # 计算标准差作为波动率
                    volatility = np.std(historical_values)
                    mean_val = np.mean(historical_values)
                    
                    # 归一化波动率
                    if mean_val > 0:
                        normalized_volatility = volatility / mean_val
                    else:
                        normalized_volatility = volatility
                    
                    domain_state.volatility[metric_name] = float(normalized_volatility)
    
    def _detect_tipping_points(self, domain_state: Any) -> None:
        """
        检测临界点距离
        
        Args:
            domain_state: 领域状态
        """
        threshold = self.config['tipping_point_threshold']
        
        for metric_name, value in domain_state.metrics.items():
            # 假设0.0-1.0的标准化范围内，0.9以上接近临界点
            distance_to_tipping = max(0.0, 1.0 - abs(value - threshold))
            domain_state.tipping_points[metric_name] = float(distance_to_tipping)
    
    def _analyze_dependencies(self, worldline: WorldLine, domain_type: DomainType,
                            domain_state: Any) -> None:
        """
        分析领域间依赖关系
        
        Args:
            worldline: 世界线
            domain_type: 领域类型
            domain_state: 领域状态
        """
        # 这里可以实现更复杂的依赖分析逻辑
        # 例如，基于历史数据计算领域间的相关性
        pass


class FeatureExtractor:
    """
    特征提取器
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化特征提取器
        
        Args:
            config: 配置信息
        """
        default_config = {
            'n_components': 5,  # PCA组件数
            'k_best_features': 10  # 选择的最佳特征数
        }
        self.config = {**default_config, **(config or {})}
    
    def extract_features(self, data: Dict[str, Any], target: Optional[np.ndarray] = None)
                       -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        从数据中提取特征
        
        Args:
            data: 输入数据
            target: 目标变量（用于监督特征选择）
            
        Returns:
            (特征矩阵, 特征名称列表, 元数据)
        """
        # 将数据转换为特征矩阵
        feature_names = []
        feature_values = []
        
        # 遍历数据并提取特征
        for key, values in data.items():
            if isinstance(values, (int, float)):
                feature_names.append(key)
                feature_values.append(values)
            elif isinstance(values, dict):
                # 递归提取嵌套字典
                for sub_key, sub_value in values.items():
                    if isinstance(sub_value, (int, float)):
                        feature_names.append(f"{key}_{sub_key}")
                        feature_values.append(sub_value)
        
        # 创建特征矩阵
        X = np.array(feature_values).reshape(1, -1) if feature_values else np.array([])
        
        metadata = {
            'original_features_count': len(feature_names),
            'extraction_time': datetime.now().timestamp()
        }
        
        # 如果有目标变量，进行特征选择
        if target is not None and len(feature_names) > self.config['k_best_features']:
            selector = SelectKBest(f_regression, k=self.config['k_best_features'])
            X_selected = selector.fit_transform(X, target)
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
            metadata['selected_features_count'] = len(selected_features)
            metadata['selected_features'] = selected_features
            
            return X_selected, selected_features, metadata
        
        # 应用PCA降维
        if X.shape[1] > self.config['n_components']:
            pca = PCA(n_components=self.config['n_components'])
            X_pca = pca.fit_transform(X)
            
            # 创建PCA特征名称
            pca_feature_names = [f'pca_component_{i}' for i in range(self.config['n_components'])]
            
            metadata['pca_applied'] = True
            metadata['explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
            metadata['cumulative_explained_variance'] = np.cumsum(pca.explained_variance_ratio_)[-1]
            
            return X_pca, pca_feature_names, metadata
        
        return X, feature_names, metadata


class DataIntegration:
    """
    数据集成类
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据集成器
        
        Args:
            config: 配置信息
        """
        self.config = config or {}
    
    def integrate_reality_datasets(self, datasets: List[RealityData]) -> RealityData:
        """
        集成多个现实数据集
        
        Args:
            datasets: 现实数据集列表
            
        Returns:
            集成后的数据集
        """
        if not datasets:
            return RealityData(name="integrated", source="multiple")
        
        # 创建集成数据集
        integrated = RealityData(
            name="integrated_reality_data",
            source="multiple_sources",
            metadata={
                'sources': [d.source for d in datasets],
                'integration_time': datetime.now().timestamp()
            }
        )
        
        # 集成数据点
        for dataset in datasets:
            for (domain_type, metric_name), data_points in dataset.data_points.items():
                # 检查是否已存在相同的数据
                existing_points = integrated.get_data_points(domain_type, metric_name)
                existing_year_value = {(y, v) for y, v in existing_points}
                
                # 添加新数据点
                for year, value in data_points:
                    if (year, value) not in existing_year_value:
                        integrated.add_data_point(domain_type, metric_name, year, value)
        
        logger.info(f"Integrated {len(datasets)} datasets, resulting in {len(integrated.data_points)} metrics")
        return integrated
    
    def align_time_series(self, data: RealityData, target_years: List[int]) -> RealityData:
        """
        将时间序列对齐到目标年份
        
        Args:
            data: 原始数据
            target_years: 目标年份列表
            
        Returns:
            对齐后的数据
        """
        aligned = RealityData(
            name=f"aligned_{data.name}",
            source=data.source,
            metadata={**data.metadata, 'aligned': True}
        )
        
        for (domain_type, metric_name), data_points in data.data_points.items():
            # 创建年份到值的映射
            year_map = {year: value for year, value in data_points}
            
            # 对于每个目标年份，找到最接近的值
            for target_year in target_years:
                if target_year in year_map:
                    # 直接使用精确匹配
                    aligned.add_data_point(domain_type, metric_name, target_year, year_map[target_year])
                else:
                    # 找到最接近的年份进行插值
                    closest_years = sorted(year_map.keys(), key=lambda y: abs(y - target_year))
                    if closest_years:
                        closest_year = closest_years[0]
                        aligned.add_data_point(domain_type, metric_name, target_year, year_map[closest_year])
        
        return aligned


class BatchProcessor:
    """
    批处理器，用于处理大量数据
    """
    
    def __init__(self, processor: DataProcessor, batch_size: int = 100):
        """
        初始化批处理器
        
        Args:
            processor: 数据处理器
            batch_size: 批大小
        """
        self.processor = processor
        self.batch_size = batch_size
    
    def process_batch(self, items: List[Any], **kwargs) -> List[Any]:
        """
        批量处理数据
        
        Args:
            items: 待处理的项目列表
            **kwargs: 传递给处理器的参数
            
        Returns:
            处理后的项目列表
        """
        results = []
        total_items = len(items)
        
        for i in range(0, total_items, self.batch_size):
            batch = items[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(total_items + self.batch_size - 1)//self.batch_size}")
            
            # 处理当前批次
            for item in batch:
                try:
                    processed_item = self.processor.process(item, **kwargs)
                    results.append(processed_item)
                except Exception as e:
                    logger.error(f"Error processing item in batch: {e}")
                    # 保留原始项目
                    results.append(item)
        
        return results


# 便捷函数
def process_reality_data(data: RealityData, config: Optional[Dict[str, Any]] = None) -> RealityData:
    """
    便捷函数：处理现实数据
    
    Args:
        data: 原始数据
        config: 配置信息
        
    Returns:
        处理后的数据
    """
    processor = RealityDataProcessor(config)
    return processor.process(data)


def process_worldline(worldline: WorldLine, config: Optional[Dict[str, Any]] = None) -> WorldLine:
    """
    便捷函数：处理世界线数据
    
    Args:
        worldline: 原始世界线
        config: 配置信息
        
    Returns:
        处理后的世界线
    """
    processor = WorldLineProcessor(config)
    return processor.process(worldline)


def integrate_datasets(datasets: List[RealityData]) -> RealityData:
    """
    便捷函数：集成多个数据集
    
    Args:
        datasets: 数据集列表
        
    Returns:
        集成后的数据集
    """
    integrator = DataIntegration()
    return integrator.integrate_reality_datasets(datasets)


# 示例用法
if __name__ == "__main__":
    # 这里可以添加示例代码
    pass