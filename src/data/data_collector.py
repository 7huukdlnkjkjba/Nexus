#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据采集模块 (Data Collector)
负责从多个数据源采集现实世界的数据
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.data_types import DomainType, RealityData
from utils.config_manager import get_config_manager

logger = get_logger(__name__)


class DataCollector:
    """
    数据采集器基类
    """
    
    def __init__(self, name: str, source: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据采集器
        
        Args:
            name: 采集器名称
            source: 数据源名称
            config: 配置信息
        """
        self.name = name
        self.source = source
        self.config = config or {}
        self.session = None
        self.rate_limit_delay = self.config.get('rate_limit_delay', 1.0)  # 速率限制延迟（秒）
        self.timeout = self.config.get('timeout', 30.0)  # 请求超时（秒）
        self.retry_count = self.config.get('retry_count', 3)  # 重试次数
        self.retry_delay = self.config.get('retry_delay', 2.0)  # 重试延迟（秒）
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def initialize(self) -> None:
        """
        初始化采集器
        """
        logger.info(f"Initializing {self.name} data collector from {self.source}")
        
        # 创建aiohttp会话
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
    
    async def close(self) -> None:
        """
        关闭采集器
        """
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"Closed {self.name} data collector")
    
    async def collect(self, **kwargs) -> RealityData:
        """
        采集数据的主方法，需要子类实现
        
        Returns:
            RealityData对象
        """
        raise NotImplementedError("Subclasses must implement collect method")
    
    async def _request_with_retry(self, url: str, method: str = 'GET',
                                 **kwargs) -> Dict[str, Any]:
        """
        带重试机制的HTTP请求
        
        Args:
            url: 请求URL
            method: 请求方法
            **kwargs: 额外的请求参数
            
        Returns:
            响应数据
        """
        for attempt in range(self.retry_count + 1):
            try:
                if not self.session:
                    await self.initialize()
                
                logger.debug(f"Making {method} request to {url}, attempt {attempt + 1}")
                
                async with getattr(self.session, method.lower())(url, **kwargs) as response:
                    if response.status == 200:
                        # 根据Content-Type决定如何解析响应
                        content_type = response.headers.get('Content-Type', '')
                        if 'json' in content_type:
                            return await response.json()
                        else:
                            # 尝试按JSON解析，失败则返回文本
                            try:
                                return await response.json()
                            except:
                                text = await response.text()
                                return {'text': text}
                    elif response.status == 429:  # 速率限制
                        retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                        logger.warning(f"Rate limited, retrying after {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                    elif response.status >= 500 or response.status == 408:  # 服务器错误或超时
                        logger.warning(f"Server error ({response.status}), retrying...")
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
                    else:
                        response.raise_for_status()
                        
            except aiohttp.ClientError as e:
                logger.error(f"HTTP error: {e}")
                if attempt < self.retry_count:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise
            except Exception as e:
                logger.error(f"Error during request: {e}")
                if attempt < self.retry_count:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
        
        raise Exception(f"Failed to get data after {self.retry_count} attempts")
    
    def _convert_to_reality_data(self, data: Dict[str, Any], **kwargs) -> RealityData:
        """
        将采集的数据转换为RealityData对象
        
        Args:
            data: 采集的数据
            **kwargs: 额外的元数据
            
        Returns:
            RealityData对象
        """
        reality_data = RealityData(
            name=self.name,
            source=self.source,
            metadata={
                'collector': self.name,
                'source': self.source,
                'collection_time': datetime.now().timestamp(),
                **kwargs
            }
        )
        return reality_data


class WorldBankCollector(DataCollector):
    """
    世界银行数据采集器
    """
    
    BASE_URL = "https://api.worldbank.org/v2"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化世界银行数据采集器
        
        Args:
            config: 配置信息
        """
        default_config = {
            'rate_limit_delay': 0.5,
            'timeout': 60.0,
            'max_items': 10000
        }
        merged_config = {**default_config, **(config or {})}
        super().__init__("world_bank", "World Bank API", merged_config)
        
        # 指标映射：领域 -> [(指标代码, 指标名称)]
        self.metric_mappings = {
            DomainType.ECONOMIC: [
                ('NY.GDP.MKTP.CD', 'GDP (current US$)'),
                ('NY.GDP.PCAP.CD', 'GDP per capita (current US$)'),
                ('NE.EXP.GNFS.CD', 'Exports of goods and services (current US$)'),
                ('NE.IMP.GNFS.CD', 'Imports of goods and services (current US$)'),
                ('FP.CPI.TOTL', 'Inflation, consumer prices (annual %)'),
                ('SL.UEM.TOTL.ZS', 'Unemployment, total (% of labor force)'),
                ('FR.INR.RINR', 'Real interest rate (%)'),
                ('BX.KLT.DINV.WD.GD.ZS', 'Foreign direct investment, net inflows (% of GDP)'),
                ('GC.DOD.TOTL.GD.ZS', 'Central government debt, total (% of GDP)')
            ],
            DomainType.SOCIAL: [
                ('SP.POP.TOTL', 'Population, total'),
                ('SP.DYN.LE00.IN', 'Life expectancy at birth, total (years)'),
                ('SE.SEC.ENRR', 'School enrollment, secondary (% net)'),
                ('SH.DYN.MORT', 'Mortality rate, under-5 (per 1,000 live births)'),
                ('SI.POV.GINI', 'Gini index'),
                ('SP.URB.TOTL.IN.ZS', 'Urban population (% of total population)')
            ],
            DomainType.ENVIRONMENTAL: [
                ('AG.LND.ARBL.HA', 'Arable land (hectares)'),
                ('AG.LND.FRST.K2', 'Forest area (sq. km)'),
                ('EN.ATM.CO2E.PC', 'CO2 emissions (metric tons per capita)'),
                ('EG.ELC.ACCS.ZS', 'Access to electricity (% of population)'),
                ('EN.ATM.PM25.MC.M3', 'PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)')
            ],
            DomainType.ENERGY: [
                ('EG.USE.PCAP.KG.OE', 'Energy use (kg of oil equivalent per capita)'),
                ('EG.ELC.RNEW.ZS', 'Electricity production from renewable sources, excluding hydroelectric (% of total)'),
                ('EG.ELC.HYRO.ZS', 'Electricity production from hydroelectric sources (% of total)'),
                ('EG.ELC.NUCL.ZS', 'Electricity production from nuclear sources (% of total)')
            ]
        }
    
    async def collect(self, countries: List[str] = None, 
                     domains: List[DomainType] = None,
                     start_year: int = 1960,
                     end_year: int = datetime.now().year - 1) -> RealityData:
        """
        从世界银行API收集数据
        
        Args:
            countries: 国家代码列表，默认为所有国家
            domains: 要收集的领域列表
            start_year: 开始年份
            end_year: 结束年份
            
        Returns:
            RealityData对象
        """
        logger.info(f"Collecting World Bank data for {start_year}-{end_year}")
        
        # 默认收集所有支持的领域
        if not domains:
            domains = list(self.metric_mappings.keys())
        
        # 默认收集所有国家
        if not countries:
            countries = ['all']
        
        # 收集数据
        reality_data = self._convert_to_reality_data({
            'countries': countries,
            'domains': [d.name for d in domains],
            'start_year': start_year,
            'end_year': end_year
        })
        
        # 收集每个领域的数据
        for domain in domains:
            if domain not in self.metric_mappings:
                logger.warning(f"No metric mappings for domain: {domain}")
                continue
            
            for indicator_code, indicator_name in self.metric_mappings[domain]:
                try:
                    await self._collect_indicator_data(
                        reality_data, domain, indicator_code, indicator_name,
                        countries, start_year, end_year
                    )
                    await asyncio.sleep(self.rate_limit_delay)
                except Exception as e:
                    logger.error(f"Error collecting {indicator_name}: {e}")
                    continue
        
        return reality_data
    
    async def _collect_indicator_data(self, reality_data: RealityData,
                                     domain: DomainType,
                                     indicator_code: str,
                                     indicator_name: str,
                                     countries: List[str],
                                     start_year: int,
                                     end_year: int) -> None:
        """
        收集单个指标的数据
        
        Args:
            reality_data: RealityData对象
            domain: 领域类型
            indicator_code: 指标代码
            indicator_name: 指标名称
            countries: 国家代码列表
            start_year: 开始年份
            end_year: 结束年份
        """
        # 构建请求URL
        country_str = ';'.join(countries)
        url = f"{self.BASE_URL}/country/{country_str}/indicator/{indicator_code}"
        
        # 构建请求参数
        params = {
            'date': f"{start_year}:{end_year}",
            'format': 'json',
            'per_page': str(self.config['max_items'])
        }
        
        # 发送请求
        response_data = await self._request_with_retry(url, params=params)
        
        # 解析响应数据
        if isinstance(response_data, list) and len(response_data) > 1:
            data_points = response_data[1]  # 第一部分是元数据，第二部分是实际数据
            
            # 按国家和年份组织数据
            country_data = {}
            for dp in data_points:
                country = dp.get('country', {}).get('value')
                year = int(dp.get('date'))
                value = dp.get('value')
                
                if country and year and value is not None:
                    if country not in country_data:
                        country_data[country] = []
                    country_data[country].append((year, value))
            
            # 添加到RealityData
            for country, points in country_data.items():
                metric_name = f"{indicator_name} ({country})"
                for year, value in points:
                    reality_data.add_data_point(domain, metric_name, year, float(value))
            
            logger.debug(f"Collected {sum(len(points) for points in country_data.values())} data points for {indicator_name}")


class IMFDataCollector(DataCollector):
    """
    IMF数据采集器
    """
    
    BASE_URL = "https://www.imf.org/external/datamapper/api/v1"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化IMF数据采集器
        
        Args:
            config: 配置信息
        """
        default_config = {
            'rate_limit_delay': 1.0,
            'timeout': 60.0
        }
        merged_config = {**default_config, **(config or {})}
        super().__init__("imf", "IMF Data API", merged_config)
        
        # 指标映射：领域 -> [(指标代码, 指标名称)]
        self.metric_mappings = {
            DomainType.ECONOMIC: [
                ('NGDPD', 'GDP, Current Prices (billions USD)'),
                ('NGDPPCH', 'GDP Growth (annual %)'),
                ('PCPIEPCH', 'Inflation, Average Consumer Prices (annual %)'),
                ('LP', 'Population (millions)'),
                ('LE', 'GDP per capita, constant prices (USD)'),
                ('GGXWDG_GDP', 'Government Gross Debt (% of GDP)'),
                ('GGSB_GDP', 'General Government Net Lending/Borrowing (% of GDP)'),
                ('BCA_GDP', 'Current Account Balance (% of GDP)'),
                ('TMG_RPCH', 'Volume of Imports of Goods (annual %)'),
                ('TXG_RPCH', 'Volume of Exports of Goods (annual %)')
            ]
        }
    
    async def collect(self, countries: List[str] = None,
                     domains: List[DomainType] = None,
                     indicators: List[str] = None,
                     start_year: int = 1980,
                     end_year: int = datetime.now().year - 1) -> RealityData:
        """
        从IMF数据API收集数据
        
        Args:
            countries: 国家代码列表
            domains: 要收集的领域列表
            indicators: 要收集的指标代码列表
            start_year: 开始年份
            end_year: 结束年份
            
        Returns:
            RealityData对象
        """
        logger.info(f"Collecting IMF data for {start_year}-{end_year}")
        
        # 默认收集所有支持的领域
        if not domains:
            domains = list(self.metric_mappings.keys())
        
        # 收集指定领域的指标
        all_indicators = []
        domain_indicators = {}
        
        if indicators:
            # 如果指定了指标，找到对应的领域
            for domain in domains:
                for code, name in self.metric_mappings.get(domain, []):
                    if code in indicators:
                        all_indicators.append(code)
                        if domain not in domain_indicators:
                            domain_indicators[domain] = []
                        domain_indicators[domain].append((code, name))
        else:
            # 否则收集所有领域的指标
            for domain in domains:
                if domain in self.metric_mappings:
                    for code, name in self.metric_mappings[domain]:
                        all_indicators.append(code)
                        if domain not in domain_indicators:
                            domain_indicators[domain] = []
                        domain_indicators[domain].append((code, name))
        
        # 构建请求URL
        indicator_str = '+'.join(all_indicators)
        url = f"{self.BASE_URL}/{indicator_str}"
        
        # 发送请求
        response_data = await self._request_with_retry(url)
        
        # 解析响应数据
        reality_data = self._convert_to_reality_data({
            'countries': countries,
            'indicators': all_indicators,
            'start_year': start_year,
            'end_year': end_year
        })
        
        # 处理数据
        for domain, domain_indicator_list in domain_indicators.items():
            for indicator_code, indicator_name in domain_indicator_list:
                if indicator_code in response_data.get('values', {}):
                    self._process_indicator_data(
                        reality_data, domain, indicator_code, indicator_name,
                        response_data['values'][indicator_code],
                        countries, start_year, end_year
                    )
        
        return reality_data
    
    def _process_indicator_data(self, reality_data: RealityData,
                               domain: DomainType,
                               indicator_code: str,
                               indicator_name: str,
                               indicator_data: Dict[str, Dict[int, float]],
                               countries: List[str],
                               start_year: int,
                               end_year: int) -> None:
        """
        处理单个指标的数据
        
        Args:
            reality_data: RealityData对象
            domain: 领域类型
            indicator_code: 指标代码
            indicator_name: 指标名称
            indicator_data: 指标数据
            countries: 国家代码列表
            start_year: 开始年份
            end_year: 结束年份
        """
        for country, year_data in indicator_data.items():
            # 如果指定了国家，检查是否在列表中
            if countries and country not in countries:
                continue
            
            for year, value in year_data.items():
                if start_year <= year <= end_year and value is not None:
                    metric_name = f"{indicator_name} ({country})"
                    reality_data.add_data_point(domain, metric_name, year, float(value))


class NASAClimateDataCollector(DataCollector):
    """
    NASA气候数据采集器
    """
    
    BASE_URL = "https://data.giss.nasa.gov/gistemp"
    GHG_DATA_URL = "https://gml.noaa.gov/ccgg/trends/data.html"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化NASA气候数据采集器
        
        Args:
            config: 配置信息
        """
        default_config = {
            'rate_limit_delay': 2.0,
            'timeout': 120.0
        }
        merged_config = {**default_config, **(config or {})}
        super().__init__("nasa_climate", "NASA GISS Climate Data", merged_config)
    
    async def collect(self, data_type: str = 'global_temperature',
                     start_year: int = 1880,
                     end_year: int = datetime.now().year) -> RealityData:
        """
        从NASA收集气候数据
        
        Args:
            data_type: 数据类型 ('global_temperature', 'co2_levels')
            start_year: 开始年份
            end_year: 结束年份
            
        Returns:
            RealityData对象
        """
        logger.info(f"Collecting NASA climate data ({data_type}) for {start_year}-{end_year}")
        
        reality_data = self._convert_to_reality_data({
            'data_type': data_type,
            'start_year': start_year,
            'end_year': end_year
        })
        
        if data_type == 'global_temperature':
            await self._collect_global_temperature(reality_data, start_year, end_year)
        elif data_type == 'co2_levels':
            await self._collect_co2_levels(reality_data, start_year, end_year)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        return reality_data
    
    async def _collect_global_temperature(self, reality_data: RealityData,
                                         start_year: int,
                                         end_year: int) -> None:
        """
        收集全球温度数据
        
        Args:
            reality_data: RealityData对象
            start_year: 开始年份
            end_year: 结束年份
        """
        # 全球温度异常数据URL
        url = f"{self.BASE_URL}/tables/v4.global.ts.csv"
        
        # 发送请求
        response = await self._request_with_retry(url, headers={'Accept': 'text/csv'})
        
        # 解析CSV数据
        if isinstance(response, dict) and 'text' in response:
            lines = response['text'].strip().split('\n')
            
            # 跳过注释行和标题行
            data_lines = []
            for line in lines:
                if not line.startswith('%') and line.strip():
                    data_lines.append(line)
            
            if data_lines:
                # 创建DataFrame
                df = pd.DataFrame([x.split(',') for x in data_lines[1:]], 
                                 columns=data_lines[0].split(','))
                
                # 处理年份和温度数据
                for _, row in df.iterrows():
                    try:
                        year = int(row[0])
                        if start_year <= year <= end_year:
                            # 全球-陆地-海洋温度异常
                            global_temp = float(row[1])
                            reality_data.add_data_point(
                                DomainType.CLIMATE,
                                "Global Temperature Anomaly (Celsius)",
                                year, global_temp
                            )
                    except (ValueError, IndexError):
                        continue
    
    async def _collect_co2_levels(self, reality_data: RealityData,
                                 start_year: int,
                                 end_year: int) -> None:
        """
        收集CO2浓度数据
        
        Args:
            reality_data: RealityData对象
            start_year: 开始年份
            end_year: 结束年份
        """
        # NOAA CO2数据URL
        url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.txt"
        
        # 发送请求
        response = await self._request_with_retry(url, headers={'Accept': 'text/plain'})
        
        # 解析数据
        if isinstance(response, dict) and 'text' in response:
            lines = response['text'].strip().split('\n')
            
            for line in lines:
                # 跳过注释行
                if line.startswith('#'):
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        year = int(parts[0])
                        co2_level = float(parts[2])
                        
                        if start_year <= year <= end_year and co2_level != -999.99:
                            reality_data.add_data_point(
                                DomainType.ENVIRONMENTAL,
                                "CO2 Concentration (ppm)",
                                year, co2_level
                            )
                except (ValueError, IndexError):
                    continue


class CustomAPICollector(DataCollector):
    """
    自定义API数据采集器
    """
    
    def __init__(self, name: str, source: str,
                 url_pattern: str,
                 domain_mapping: Dict[str, DomainType],
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化自定义API采集器
        
        Args:
            name: 采集器名称
            source: 数据源名称
            url_pattern: URL模式，支持{placeholder}格式
            domain_mapping: 指标到领域的映射
            config: 配置信息
        """
        super().__init__(name, source, config)
        self.url_pattern = url_pattern
        self.domain_mapping = domain_mapping
    
    async def collect(self, url_params: Dict[str, Any] = None,
                     data_extractor: Callable = None,
                     **kwargs) -> RealityData:
        """
        从自定义API收集数据
        
        Args:
            url_params: URL参数，用于填充URL模式
            data_extractor: 数据提取函数
            **kwargs: 额外的请求参数
            
        Returns:
            RealityData对象
        """
        # 构建URL
        url = self.url_pattern
        if url_params:
            url = url.format(**url_params)
        
        # 发送请求
        response_data = await self._request_with_retry(url, **kwargs)
        
        # 创建RealityData对象
        reality_data = self._convert_to_reality_data({
            'url': url,
            'params': url_params
        })
        
        # 使用自定义提取器或默认提取逻辑
        if data_extractor:
            data_extractor(response_data, reality_data, self.domain_mapping)
        else:
            self._default_data_extractor(response_data, reality_data)
        
        return reality_data
    
    def _default_data_extractor(self, data: Dict[str, Any],
                               reality_data: RealityData) -> None:
        """
        默认的数据提取逻辑
        
        Args:
            data: 原始数据
            reality_data: RealityData对象
        """
        # 这是一个简单的默认实现，应该根据具体API的响应格式进行重写
        pass


class DataCollectorManager:
    """
    数据采集器管理器
    """
    
    def __init__(self):
        """
        初始化数据采集器管理器
        """
        self.collectors: Dict[str, DataCollector] = {}
        self.config_manager = get_config_manager()
    
    def register_collector(self, name: str, collector: DataCollector) -> None:
        """
        注册数据采集器
        
        Args:
            name: 采集器名称
            collector: 数据采集器实例
        """
        self.collectors[name] = collector
        logger.info(f"Registered data collector: {name}")
    
    def get_collector(self, name: str) -> Optional[DataCollector]:
        """
        获取数据采集器
        
        Args:
            name: 采集器名称
            
        Returns:
            数据采集器实例
        """
        return self.collectors.get(name)
    
    async def collect_data(self, collector_name: str, **kwargs) -> RealityData:
        """
        使用指定的采集器收集数据
        
        Args:
            collector_name: 采集器名称
            **kwargs: 传递给采集器的参数
            
        Returns:
            RealityData对象
        """
        if collector_name not in self.collectors:
            raise ValueError(f"Collector not found: {collector_name}")
        
        collector = self.collectors[collector_name]
        return await collector.collect(**kwargs)
    
    async def collect_all(self, collectors: List[str] = None,
                         params_map: Dict[str, Dict[str, Any]] = None) -> Dict[str, RealityData]:
        """
        收集所有或指定的采集器的数据
        
        Args:
            collectors: 要使用的采集器名称列表
            params_map: 采集器名称到参数的映射
            
        Returns:
            采集器名称到RealityData的映射
        """
        if not collectors:
            collectors = list(self.collectors.keys())
        
        if not params_map:
            params_map = {}
        
        results = {}
        
        # 并发收集数据
        tasks = []
        for name in collectors:
            if name in self.collectors:
                params = params_map.get(name, {})
                tasks.append((name, self.collect_data(name, **params)))
        
        # 执行并收集结果
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Error collecting data from {name}: {e}")
                results[name] = None
        
        return results
    
    def initialize_default_collectors(self) -> None:
        """
        初始化默认的数据采集器
        """
        # 从配置中读取API密钥等信息
        api_config = self.config_manager.get('api_keys', {})
        
        # 初始化世界银行采集器
        world_bank_config = api_config.get('world_bank', {})
        world_bank_collector = WorldBankCollector(world_bank_config)
        self.register_collector('world_bank', world_bank_collector)
        
        # 初始化IMF采集器
        imf_config = api_config.get('imf', {})
        imf_collector = IMFDataCollector(imf_config)
        self.register_collector('imf', imf_collector)
        
        # 初始化NASA气候数据采集器
        nasa_config = api_config.get('nasa', {})
        nasa_collector = NASAClimateDataCollector(nasa_config)
        self.register_collector('nasa_climate', nasa_collector)
        
        logger.info("Initialized default data collectors")
    
    async def close_all(self) -> None:
        """
        关闭所有采集器
        """
        for collector in self.collectors.values():
            await collector.close()


# 全局数据采集器管理器实例
_data_collector_manager = None

def init_data_collector_manager() -> DataCollectorManager:
    """
    初始化全局数据采集器管理器
    
    Returns:
        数据采集器管理器实例
    """
    global _data_collector_manager
    _data_collector_manager = DataCollectorManager()
    _data_collector_manager.initialize_default_collectors()
    return _data_collector_manager

def get_data_collector_manager() -> DataCollectorManager:
    """
    获取全局数据采集器管理器实例
    
    Returns:
        数据采集器管理器实例
    """
    global _data_collector_manager
    if _data_collector_manager is None:
        init_data_collector_manager()
    return _data_collector_manager


# 便捷函数
async def collect_world_bank_data(**kwargs) -> RealityData:
    """
    便捷函数：收集世界银行数据
    
    Args:
        **kwargs: 传递给collect方法的参数
        
    Returns:
        RealityData对象
    """
    manager = get_data_collector_manager()
    return await manager.collect_data('world_bank', **kwargs)


async def collect_imf_data(**kwargs) -> RealityData:
    """
    便捷函数：收集IMF数据
    
    Args:
        **kwargs: 传递给collect方法的参数
        
    Returns:
        RealityData对象
    """
    manager = get_data_collector_manager()
    return await manager.collect_data('imf', **kwargs)


async def collect_climate_data(**kwargs) -> RealityData:
    """
    便捷函数：收集气候数据
    
    Args:
        **kwargs: 传递给collect方法的参数
        
    Returns:
        RealityData对象
    """
    manager = get_data_collector_manager()
    return await manager.collect_data('nasa_climate', **kwargs)


# 示例用法演示
async def example_usage():
    """
    示例用法
    """
    # 创建数据采集器管理器
    manager = get_data_collector_manager()
    
    # 收集世界银行数据（经济指标）
    try:
        wb_data = await collect_world_bank_data(
            countries=['USA', 'CHN', 'IND'],
            domains=[DomainType.ECONOMIC],
            start_year=2000,
            end_year=2020
        )
        print(f"World Bank data collected: {len(wb_data.data_points)} metrics")
    except Exception as e:
        print(f"Error collecting World Bank data: {e}")
    
    # 收集IMF数据
    try:
        imf_data = await collect_imf_data(
            countries=['USA', 'CHN', 'IND'],
            start_year=2000,
            end_year=2020
        )
        print(f"IMF data collected: {len(imf_data.data_points)} metrics")
    except Exception as e:
        print(f"Error collecting IMF data: {e}")
    
    # 收集气候数据
    try:
        climate_data = await collect_climate_data(
            data_type='global_temperature',
            start_year=1900,
            end_year=2020
        )
        print(f"Climate data collected: {len(climate_data.data_points)} metrics")
    except Exception as e:
        print(f"Error collecting climate data: {e}")
    
    # 关闭所有采集器
    await manager.close_all()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())