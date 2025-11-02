#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nexus - 全球博弈模拟与策略超前推演引擎
技术模型模块
负责模拟全球技术发展和创新系统的演化
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import random

from ..utils.logger import get_logger

class TechnologicalModel:
    """技术模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化技术模型
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("TechnologicalModel")
        self.config = config
        
        # 模型参数
        self.volatility = config.get("technological_volatility", 0.03)  # 技术发展波动性
        self.base_progress_rate = config.get("base_progress_rate", 0.05)  # 基础技术进步率
        self.breakthrough_probability = config.get("breakthrough_probability", 0.02)  # 技术突破概率
        self.diffusion_rate = config.get("technology_diffusion_rate", 0.15)  # 技术扩散率
        
        # 国家参数映射
        self.country_parameters = config.get("country_parameters", {})
        
        # 领域间影响权重
        self.cross_impact_weights = {
            "economic": config.get("economic_impact_weight", 0.35),
            "political": config.get("political_impact_weight", 0.25),
            "climate": config.get("climate_impact_weight", 0.3)
        }
        
        # 技术领域定义
        self.tech_domains = config.get("tech_domains", [
            "artificial_intelligence",
            "renewable_energy",
            "quantum_computing",
            "biotechnology",
            "space_exploration",
            "robotics",
            "nanotechnology",
            "blockchain",
            "advanced_materials",
            "digital_infrastructure"
        ])
        
        # 技术发展水平阈值（0-100）
        self.tech_level_thresholds = {
            "early_research": 20,
            "prototyping": 40,
            "early_adoption": 60,
            "widespread": 80,
            "mature": 95
        }
        
        self.logger.info("技术模型初始化完成")
    
    def evolve(self, technological_state: Dict[str, Any], 
              current_time: datetime, 
              random_state: np.random.RandomState) -> Dict[str, Any]:
        """
        演化技术状态
        
        Args:
            technological_state: 当前技术状态
            current_time: 当前时间
            random_state: 随机数生成器
            
        Returns:
            更新后的技术状态
        """
        # 复制状态以避免修改原始数据
        new_state = technological_state.copy()
        
        # 演化各领域技术水平
        for domain in self.tech_domains:
            if domain in new_state:
                new_state[domain] = self._evolve_tech_domain(
                    domain,
                    new_state[domain],
                    new_state.get("investment", {}).get(domain, {}),
                    new_state.get("breakthroughs", {}).get(domain, []),
                    random_state
                )
        
        # 演化技术投资
        if "investment" in new_state:
            new_state["investment"] = self._evolve_investment(new_state["investment"], random_state)
        
        # 演化技术扩散
        if "diffusion" in new_state:
            new_state["diffusion"] = self._evolve_diffusion(new_state["diffusion"], new_state, random_state)
        
        # 演化技术突破记录
        if "breakthroughs" not in new_state:
            new_state["breakthroughs"] = {domain: [] for domain in self.tech_domains}
        
        # 可能生成技术突破
        new_breakthroughs = self._generate_breakthroughs(new_state, random_state, current_time)
        for domain, breakthroughs in new_breakthroughs.items():
            if domain not in new_state["breakthroughs"]:
                new_state["breakthroughs"][domain] = []
            new_state["breakthroughs"][domain].extend(breakthroughs)
        
        # 演化专利活动
        if "patents" in new_state:
            new_state["patents"] = self._evolve_patents(new_state["patents"], new_state, random_state)
        
        # 更新时间戳
        new_state["last_updated"] = current_time.isoformat()
        
        return new_state
    
    def _evolve_tech_domain(self, domain: str, 
                           current_level: float,
                           investment: Dict[str, float],
                           past_breakthroughs: List[Dict[str, Any]],
                           random_state: np.random.RandomState) -> float:
        """
        演化特定技术领域的发展水平
        
        Args:
            domain: 技术领域
            current_level: 当前发展水平
            investment: 该领域的投资情况
            past_breakthroughs: 过去的技术突破
            random_state: 随机数生成器
            
        Returns:
            更新后的发展水平
        """
        # 基础进步率
        progress_rate = self.base_progress_rate
        
        # 投资影响（投资越多，进步越快）
        total_investment = sum(investment.values()) if investment else 0
        investment_boost = 0.02 * np.log1p(total_investment)  # 对数增长，避免过快
        
        # 突破性技术的长期影响递减
        breakthrough_boost = 0.0
        for breakthrough in past_breakthroughs:
            # 假设突破的影响随时间递减
            age = datetime.now().year - datetime.fromisoformat(breakthrough.get("date", datetime.now().isoformat())).year
            if age < 5:  # 5年内的突破仍有显著影响
                breakthrough_boost += 0.03 / (1 + age)  # 影响随时间递减
        
        # 随机波动
        random_component = random_state.normal(0, self.volatility)
        
        # 接近上限时，进步速度减慢（技术成熟效应）
        maturity_factor = (100 - current_level) / 100  # 越接近100，进步越慢
        
        # 计算新的技术水平
        new_level = current_level + (progress_rate + investment_boost + breakthrough_boost + random_component) * maturity_factor
        
        # 限制在0-100范围内
        new_level = max(0, min(100, new_level))
        
        return new_level
    
    def _evolve_investment(self, investment: Dict[str, Dict[str, float]],
                          random_state: np.random.RandomState) -> Dict[str, Dict[str, float]]:
        """
        演化技术投资情况
        
        Args:
            investment: 当前投资情况 {domain: {country: amount}}
            random_state: 随机数生成器
            
        Returns:
            更新后的投资情况
        """
        updated_investment = {}
        
        # 主要投资国家
        major_investors = ["US", "China", "EU", "Japan", "South Korea", "Israel", "Germany", "UK"]
        
        for domain in self.tech_domains:
            updated_investment[domain] = {}
            
            # 如果该领域已有投资记录，基于此更新
            if domain in investment:
                for country, amount in investment[domain].items():
                    # 投资波动
                    change_factor = random_state.normal(1.05, 0.2)  # 平均5%增长，20%波动
                    new_amount = amount * change_factor
                    updated_investment[domain][country] = max(0, new_amount)  # 确保不为负
            
            # 新国家可能进入投资
            for country in major_investors:
                if country not in updated_investment[domain]:
                    # 小概率开始投资
                    if random_state.random() < 0.05:
                        # 初始投资金额
                        initial_investment = random_state.uniform(10, 100)  # 假设单位为十亿美元
                        updated_investment[domain][country] = initial_investment
        
        return updated_investment
    
    def _evolve_diffusion(self, diffusion: Dict[str, Dict[str, float]],
                         tech_state: Dict[str, Any],
                         random_state: np.random.RandomState) -> Dict[str, Dict[str, float]]:
        """
        演化技术扩散情况
        
        Args:
            diffusion: 当前扩散情况 {domain: {country: adoption_rate}}
            tech_state: 当前技术状态
            random_state: 随机数生成器
            
        Returns:
            更新后的扩散情况
        """
        updated_diffusion = {}
        
        # 主要国家
        major_countries = ["US", "China", "EU", "Japan", "South Korea", "India", "Brazil", "Russia"]
        
        for domain in self.tech_domains:
            updated_diffusion[domain] = {}
            
            # 技术领域的全球发展水平
            global_tech_level = tech_state.get(domain, 0)
            
            for country in major_countries:
                # 当前采用率
                current_adoption = diffusion.get(domain, {}).get(country, 0)
                
                # 理想采用率（基于技术水平）
                # 技术越成熟，潜在采用率越高
                target_adoption = min(100, global_tech_level * random_state.uniform(0.8, 1.2))
                
                # 扩散速度受国家发展水平影响
                country_factor = self._get_country_tech_factor(country)
                
                # 计算扩散增量（趋向目标采用率）
                diffusion_increment = self.diffusion_rate * country_factor * (target_adoption - current_adoption)
                
                # 随机波动
                random_factor = random_state.normal(1.0, 0.2)
                
                # 新的采用率
                new_adoption = current_adoption + diffusion_increment * random_factor
                
                # 限制在0-100范围内
                new_adoption = max(0, min(100, new_adoption))
                
                updated_diffusion[domain][country] = new_adoption
        
        return updated_diffusion
    
    def _generate_breakthroughs(self, tech_state: Dict[str, Any],
                               random_state: np.random.RandomState,
                               current_time: datetime) -> Dict[str, List[Dict[str, Any]]]:
        """
        生成技术突破
        
        Args:
            tech_state: 当前技术状态
            random_state: 随机数生成器
            current_time: 当前时间
            
        Returns:
            各领域的技术突破
        """
        new_breakthroughs = {domain: [] for domain in self.tech_domains}
        
        # 突破性发现的基础描述
        breakthrough_templates = {
            "artificial_intelligence": [
                "大型语言模型在因果推理方面取得重大突破",
                "多模态AI系统实现了人类级别的视觉-语言理解",
                "自主AI代理在复杂环境中展现出前所未有的适应性",
                "AI系统在无监督学习领域实现了质的飞跃",
                "神经符号推理技术使得AI能够解释其决策过程"
            ],
            "renewable_energy": [
                "新型太阳能电池效率突破40%大关",
                "大规模储能技术取得革命性进展",
                "聚变能源研究实现了能量增益大于1",
                "氢能源基础设施取得突破性进展",
                "新型风力发电技术显著提高了发电效率"
            ],
            "quantum_computing": [
                "量子纠错技术取得重大进展，大幅延长量子相干时间",
                "实现了具有数百个逻辑量子比特的量子计算机",
                "量子算法在特定问题上展现出指数级优势",
                "常温量子计算取得突破性进展",
                "量子网络实现了远距离安全量子通信"
            ],
            "biotechnology": [
                "CRISPR技术在基因编辑精确性方面取得突破",
                "合成生物学成功构建了功能性人工细胞",
                "mRNA技术在治疗多种疾病方面展现出巨大潜力",
                "脑机接口技术允许直接思维控制外部设备",
                "再生医学实现了复杂器官的体外培育"
            ],
            "space_exploration": [
                "可重复使用火箭技术取得重大进展",
                "火星殖民地基础设施建设取得突破性进展",
                "深空探测器发现了外星生命存在的潜在证据",
                "太空电梯技术取得理论突破",
                "小行星采矿技术实现商业化应用"
            ]
            # 可以继续为其他领域添加模板
        }
        
        for domain in self.tech_domains:
            # 当前技术水平
            current_level = tech_state.get(domain, 0)
            
            # 技术水平越高，突破概率越大（但接近100时突破越难）
            # 倒U型曲线：中等水平时突破概率最高
            base_probability = self.breakthrough_probability
            
            # 技术发展中期突破概率较高
            if 30 < current_level < 80:
                base_probability *= 1.5
            
            # 判断是否发生突破
            if random_state.random() < base_probability:
                # 主要突破国家
                leading_countries = self._get_leading_countries(domain, tech_state.get("investment", {}), random_state)
                
                # 突破严重程度（1-5）
                severity = random_state.randint(1, 6)
                
                # 获取突破描述
                templates = breakthrough_templates.get(domain, [f"{domain} 领域取得重大技术突破"])
                description = random_state.choice(templates)
                
                # 创建突破记录
                breakthrough = {
                    "id": f"breakthrough_{domain}_{int(current_time.timestamp())}_{random_state.randint(1000, 9999)}",
                    "description": description,
                    "severity": severity,
                    "leading_countries": leading_countries,
                    "date": current_time.isoformat(),
                    "impact": self._calculate_breakthrough_impact(severity, domain)
                }
                
                new_breakthroughs[domain].append(breakthrough)
                self.logger.info(f"技术突破: {description} (领域: {domain}, 严重度: {severity})")
        
        return new_breakthroughs
    
    def _evolve_patents(self, patents: Dict[str, Dict[str, int]],
                       tech_state: Dict[str, Any],
                       random_state: np.random.RandomState) -> Dict[str, Dict[str, int]]:
        """
        演化专利活动
        
        Args:
            patents: 当前专利情况 {domain: {country: count}}
            tech_state: 当前技术状态
            random_state: 随机数生成器
            
        Returns:
            更新后的专利情况
        """
        updated_patents = {}
        
        # 主要专利申请国家
        major_patent_countries = ["US", "China", "Japan", "Korea", "Germany", "Taiwan", "France", "UK"]
        
        for domain in self.tech_domains:
            updated_patents[domain] = {}
            
            # 当前技术水平影响专利数量
            tech_level = tech_state.get(domain, 0)
            base_patent_rate = tech_level * 2  # 技术水平越高，专利申请越多
            
            for country in major_patent_countries:
                # 国家技术能力因子
                country_factor = self._get_country_tech_factor(country)
                
                # 投资影响
                investment = tech_state.get("investment", {}).get(domain, {}).get(country, 0)
                investment_factor = 1.0 + 0.01 * investment
                
                # 计算新增专利数
                expected_patents = base_patent_rate * country_factor * investment_factor
                new_patents = int(random_state.poisson(expected_patents))
                
                # 累加历史专利数
                existing_patents = patents.get(domain, {}).get(country, 0)
                updated_patents[domain][country] = existing_patents + new_patents
        
        return updated_patents
    
    def _get_country_tech_factor(self, country: str) -> float:
        """
        获取国家技术能力因子
        
        Args:
            country: 国家名称
            
        Returns:
            技术能力因子
        """
        # 预定义主要国家的技术能力因子
        tech_factors = {
            "US": 1.0,
            "China": 0.95,
            "Japan": 0.85,
            "South Korea": 0.8,
            "Germany": 0.8,
            "Israel": 0.75,
            "Switzerland": 0.7,
            "Singapore": 0.7,
            "Taiwan": 0.7,
            "EU": 0.85,
            "UK": 0.75,
            "France": 0.7
        }
        
        # 如果没有预定义，使用默认值
        if country not in tech_factors:
            return 0.5
        
        # 从配置中获取可能的自定义因子
        custom_factor = self.country_parameters.get(country, {}).get("tech_factor")
        if custom_factor is not None:
            return custom_factor
        
        return tech_factors.get(country, 0.5)
    
    def _get_leading_countries(self, domain: str, 
                              investment: Dict[str, Dict[str, float]],
                              random_state: np.random.RandomState) -> List[str]:
        """
        获取领域领先国家
        
        Args:
            domain: 技术领域
            investment: 投资情况
            random_state: 随机数生成器
            
        Returns:
            领先国家列表
        """
        if domain not in investment:
            # 如果没有投资数据，返回随机的主要技术国家
            major_tech_countries = ["US", "China", "Japan", "Germany", "South Korea"]
            return random_state.choice(major_tech_countries, size=min(2, len(major_tech_countries)), replace=False).tolist()
        
        # 基于投资金额排序
        country_investments = investment[domain]
        sorted_countries = sorted(country_investments.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前几个投资最多的国家
        leading_countries = [country for country, _ in sorted_countries[:3]]
        
        # 随机选择1-2个国家作为主要突破者
        num_to_select = random_state.randint(1, 3)
        return random_state.choice(leading_countries, size=min(num_to_select, len(leading_countries)), replace=False).tolist()
    
    def _calculate_breakthrough_impact(self, severity: int, domain: str) -> Dict[str, float]:
        """
        计算技术突破的影响
        
        Args:
            severity: 严重程度（1-5）
            domain: 技术领域
            
        Returns:
            各方面的影响程度
        """
        # 基础影响因子
        base_impact = severity / 5.0
        
        # 不同领域对不同方面的影响权重
        impact_weights = {
            "economic": {
                "renewable_energy": 1.2,
                "artificial_intelligence": 1.1,
                "quantum_computing": 1.0,
                "biotechnology": 0.9,
                "advanced_materials": 0.8
            },
            "political": {
                "space_exploration": 1.1,
                "artificial_intelligence": 1.0,
                "quantum_computing": 0.9
            },
            "technological": {
                "artificial_intelligence": 1.2,
                "quantum_computing": 1.2,
                "biotechnology": 1.1,
                "advanced_materials": 1.0
            },
            "climate": {
                "renewable_energy": 1.3,
                "advanced_materials": 0.9,
                "artificial_intelligence": 0.8
            }
        }
        
        impact = {}
        for area in impact_weights:
            # 获取该领域在该方面的权重，如果没有则使用默认值
            weight = impact_weights[area].get(domain, 0.7)
            # 计算影响程度
            impact_value = base_impact * weight * 100  # 转换为百分比
            impact[area] = min(100, impact_value)  # 限制最大值
        
        return impact
    
    def calculate_technology_readiness_level(self, tech_state: Dict[str, Any]) -> Dict[str, str]:
        """
        计算各技术领域的准备度水平
        
        Args:
            tech_state: 技术状态
            
        Returns:
            各领域的技术准备度水平
        """
        readiness_levels = {}
        
        for domain in self.tech_domains:
            if domain in tech_state:
                level = tech_state[domain]
                
                if level < self.tech_level_thresholds["early_research"]:
                    readiness = "早期研究"
                elif level < self.tech_level_thresholds["prototyping"]:
                    readiness = "原型开发"
                elif level < self.tech_level_thresholds["early_adoption"]:
                    readiness = "早期应用"
                elif level < self.tech_level_thresholds["widespread"]:
                    readiness = "广泛应用"
                elif level < self.tech_level_thresholds["mature"]:
                    readiness = "高度成熟"
                else:
                    readiness = "技术饱和"
                
                readiness_levels[domain] = readiness
        
        return readiness_levels
    
    def get_global_innovation_index(self, tech_state: Dict[str, Any]) -> Dict[str, float]:
        """
        计算全球创新指数
        
        Args:
            tech_state: 技术状态
            
        Returns:
            全球和各领域的创新指数
        """
        indices = {}
        
        # 计算各领域创新指数
        for domain in self.tech_domains:
            if domain in tech_state:
                # 基础指数基于技术水平
                base_index = tech_state[domain]
                
                # 考虑近期突破的影响
                recent_breakthroughs = 0
                if "breakthroughs" in tech_state and domain in tech_state["breakthroughs"]:
                    # 计算最近2年内的突破数量
                    for breakthrough in tech_state["breakthroughs"][domain]:
                        try:
                            breakthrough_year = datetime.fromisoformat(breakthrough.get("date", 
                                                                                      datetime.now().isoformat())).year
                            if breakthrough_year >= datetime.now().year - 2:
                                recent_breakthroughs += 1
                        except:
                            pass
                
                # 突破带来的指数提升
                breakthrough_bonus = recent_breakthroughs * 5  # 每个近期突破加5分
                
                indices[domain] = min(100, base_index + breakthrough_bonus)
        
        # 计算全球创新指数（各领域平均值）
        if indices:
            indices["global"] = sum(indices.values()) / len(indices)
        else:
            indices["global"] = 0.0
        
        return indices

# 测试代码
if __name__ == "__main__":
    # 简单的测试配置
    config = {
        "technological_volatility": 0.03,
        "base_progress_rate": 0.05,
        "breakthrough_probability": 0.02,
        "technology_diffusion_rate": 0.15,
        "country_parameters": {
            "US": {
                "tech_factor": 1.0,
                "research_funding": 100
            },
            "China": {
                "tech_factor": 0.95,
                "research_funding": 90
            }
        }
    }
    
    # 创建技术模型
    tech_model = TechnologicalModel(config)
    
    # 创建初始技术状态
    initial_state = {
        "artificial_intelligence": 65.0,
        "renewable_energy": 55.0,
        "quantum_computing": 35.0,
        "biotechnology": 60.0,
        "space_exploration": 40.0,
        "robotics": 50.0,
        "nanotechnology": 45.0,
        "blockchain": 30.0,
        "advanced_materials": 52.0,
        "digital_infrastructure": 68.0,
        "investment": {
            "artificial_intelligence": {
                "US": 50.0,
                "China": 45.0,
                "EU": 25.0,
                "Japan": 15.0
            },
            "renewable_energy": {
                "US": 30.0,
                "China": 55.0,
                "EU": 40.0,
                "Germany": 20.0
            },
            "quantum_computing": {
                "US": 40.0,
                "China": 35.0,
                "EU": 20.0,
                "Japan": 15.0
            }
        },
        "diffusion": {
            "artificial_intelligence": {
                "US": 70.0,
                "China": 65.0,
                "EU": 60.0,
                "Japan": 75.0
            },
            "renewable_energy": {
                "US": 45.0,
                "China": 50.0,
                "EU": 60.0,
                "Germany": 65.0
            }
        },
        "breakthroughs": {
            "artificial_intelligence": [
                {
                    "id": "ai_breakthrough_1",
                    "description": "大型语言模型在上下文理解方面取得重大突破",
                    "severity": 4,
                    "leading_countries": ["US", "China"],
                    "date": (datetime.now() - timedelta(days=180)).isoformat()
                }
            ],
            "renewable_energy": [
                {
                    "id": "energy_breakthrough_1",
                    "description": "新型电池技术使得储能成本降低50%",
                    "severity": 5,
                    "leading_countries": ["US", "Japan"],
                    "date": (datetime.now() - timedelta(days=90)).isoformat()
                }
            ]
        },
        "patents": {
            "artificial_intelligence": {
                "US": 15000,
                "China": 18000,
                "Japan": 8000,
                "Korea": 5000
            },
            "renewable_energy": {
                "China": 20000,
                "US": 12000,
                "EU": 10000,
                "Germany": 7000
            }
        }
    }
    
    # 模拟演化
    current_time = datetime.now()
    random_state = np.random.RandomState(42)
    
    print("初始技术状态:")
    print(f"人工智能水平: {initial_state['artificial_intelligence']}")
    print(f"可再生能源水平: {initial_state['renewable_energy']}")
    print(f"量子计算水平: {initial_state['quantum_computing']}")
    print()
    
    # 演化10步
    for step in range(10):
        next_time = current_time + timedelta(days=30)  # 假设每月一步
        new_state = tech_model.evolve(initial_state, next_time, random_state)
        initial_state = new_state
    
    print("演化后技术状态:")
    print(f"人工智能水平: {initial_state['artificial_intelligence']}")
    print(f"可再生能源水平: {initial_state['renewable_energy']}")
    print(f"量子计算水平: {initial_state['quantum_computing']}")
    print()
    
    # 检查新增技术突破
    new_ai_breakthroughs = [b for b in initial_state['breakthroughs']['artificial_intelligence'] 
                          if b['id'] != 'ai_breakthrough_1']
    if new_ai_breakthroughs:
        print("人工智能领域新增突破:")
        for breakthrough in new_ai_breakthroughs:
            print(f"- {breakthrough['description']} (严重度: {breakthrough['severity']})")
    
    # 计算技术准备度
    readiness_levels = tech_model.calculate_technology_readiness_level(initial_state)
    print(f"\n技术准备度水平:")
    for domain, level in readiness_levels.items():
        print(f"{domain}: {level}")
    
    # 计算创新指数
    innovation_indices = tech_model.get_global_innovation_index(initial_state)
    print(f"\n创新指数:")
    for domain, index in innovation_indices.items():
        print(f"{domain}: {index:.2f}")