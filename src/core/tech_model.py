import numpy as np
from typing import Dict, List, Any, Optional
import datetime
import logging

class TechModel:
    """
    技术领域模型 - 负责模拟全球技术发展、创新扩散和技术突破
    
    核心功能:
    - 技术状态演化
    - 创新扩散
    - 技术突破事件
    - 技术依赖关系
    - 技术封锁和制裁影响
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化技术模型
        
        Args:
            config: 配置参数
        """
        self.logger = logging.getLogger(__name__)
        
        # 默认配置
        self.default_config = {
            "volatility": 0.1,  # 技术发展波动性
            "innovation_rate": 0.02,  # 基础创新速率
            "diffusion_rate": 0.05,  # 技术扩散速率
            "breakthrough_probability": 0.005,  # 技术突破概率
            "dependency_impact": 0.1,  # 技术依赖影响系数
            "sanction_impact": 0.3,  # 制裁影响系数
            "max_tech_level": 100.0,  # 最大技术水平
            "min_tech_level": 0.0,  # 最小技术水平
            "tech_sectors": [
                "artificial_intelligence",
                "quantum_computing",
                "biotechnology",
                "renewable_energy",
                "advanced_materials",
                "space_technology",
                "semiconductors",
                "cybersecurity",
                "blockchain",
                "robotics"
            ],
            "dependency_matrix": {},  # 技术依赖矩阵
            "sanction_targets": {},  # 制裁目标
        }
        
        # 合并配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # 提取配置参数
        self.volatility = self.config["volatility"]
        self.innovation_rate = self.config["innovation_rate"]
        self.diffusion_rate = self.config["diffusion_rate"]
        self.breakthrough_probability = self.config["breakthrough_probability"]
        self.dependency_impact = self.config["dependency_impact"]
        self.sanction_impact = self.config["sanction_impact"]
        self.max_tech_level = self.config["max_tech_level"]
        self.min_tech_level = self.config["min_tech_level"]
        self.tech_sectors = self.config["tech_sectors"]
        self.dependency_matrix = self.config["dependency_matrix"]
        self.sanction_targets = self.config["sanction_targets"]
        
        # 初始化依赖矩阵
        if not self.dependency_matrix:
            self._initialize_dependency_matrix()
        
        self.logger.info("技术模型初始化完成")
    
    def _initialize_dependency_matrix(self):
        """
        初始化技术依赖矩阵
        """
        # 定义基本的技术依赖关系
        self.dependency_matrix = {
            "artificial_intelligence": ["semiconductors", "cybersecurity"],
            "quantum_computing": ["advanced_materials", "semiconductors"],
            "biotechnology": ["advanced_materials"],
            "renewable_energy": ["advanced_materials"],
            "advanced_materials": [],
            "space_technology": ["advanced_materials", "semiconductors", "robotics"],
            "semiconductors": [],
            "cybersecurity": ["artificial_intelligence", "semiconductors"],
            "blockchain": ["cryptography", "cybersecurity"],
            "robotics": ["artificial_intelligence", "advanced_materials", "semiconductors"]
        }
        self.logger.info("技术依赖矩阵初始化完成")
    
    def evolve(self, tech_state: Dict[str, Dict[str, float]],
              global_conditions: Dict[str, Any],
              random_state: np.random.RandomState) -> Dict[str, Dict[str, float]]:
        """
        演化技术状态
        
        Args:
            tech_state: 当前技术状态 {country: {tech_sector: level}}
            global_conditions: 全球条件
            random_state: 随机数生成器
            
        Returns:
            更新后的技术状态
        """
        updated_tech_state = {}
        
        # 复制现有状态作为基础
        for country, tech_levels in tech_state.items():
            updated_tech_state[country] = tech_levels.copy()
        
        # 演化每个国家的技术状态
        for country, tech_levels in updated_tech_state.items():
            # 检查是否受到制裁
            is_sanctioned = self._is_country_sanctioned(country)
            
            # 演化每个技术领域
            for tech_sector in self.tech_sectors:
                if tech_sector not in tech_levels:
                    tech_levels[tech_sector] = 0.0
                
                # 基础技术发展
                base_growth = self._calculate_base_growth(
                    country, tech_sector, tech_levels, global_conditions, random_state
                )
                
                # 技术依赖影响
                dependency_impact = self._calculate_dependency_impact(
                    tech_sector, tech_levels
                )
                
                # 制裁影响
                sanction_penalty = 0.0
                if is_sanctioned and tech_sector in self.sanction_targets.get(country, []):
                    sanction_penalty = -self.sanction_impact * tech_levels[tech_sector] * random_state.random() * 0.5
                
                # 技术突破
                breakthrough_boost = self._check_for_breakthrough(
                    country, tech_sector, tech_levels, random_state
                )
                
                # 技术扩散
                diffusion_boost = self._calculate_diffusion_impact(
                    country, tech_sector, tech_state, random_state
                )
                
                # 计算总变化
                total_change = base_growth + dependency_impact + sanction_penalty + breakthrough_boost + diffusion_boost
                
                # 更新技术水平
                new_level = tech_levels[tech_sector] + total_change
                tech_levels[tech_sector] = max(self.min_tech_level, min(self.max_tech_level, new_level))
        
        self.logger.debug(f"技术状态演化完成，涉及 {len(updated_tech_state)} 个国家")
        return updated_tech_state
    
    def _calculate_base_growth(self, country: str, tech_sector: str,
                              tech_levels: Dict[str, float],
                              global_conditions: Dict[str, Any],
                              random_state: np.random.RandomState) -> float:
        """
        计算基础技术发展
        
        Args:
            country: 国家
            tech_sector: 技术领域
            tech_levels: 技术水平
            global_conditions: 全球条件
            random_state: 随机数生成器
            
        Returns:
            基础增长率
        """
        # 基础创新率
        base_rate = self.innovation_rate
        
        # 技术水平越高，增长越慢（边际递减）
        current_level = tech_levels.get(tech_sector, 0.0)
        diminishing_factor = 1.0 - (current_level / self.max_tech_level)
        
        # 随机波动
        volatility_factor = random_state.normal(1.0, self.volatility)
        
        # 计算基础增长
        base_growth = base_rate * diminishing_factor * volatility_factor
        
        # 全球条件影响
        if global_conditions:
            # 考虑经济状况影响
            economic_health = global_conditions.get("economic_health", {}).get(country, 1.0)
            base_growth *= economic_health
            
            # 考虑政治稳定性影响
            political_stability = global_conditions.get("political_stability", {}).get(country, 1.0)
            base_growth *= political_stability
        
        return base_growth
    
    def _calculate_dependency_impact(self, tech_sector: str,
                                    tech_levels: Dict[str, float]) -> float:
        """
        计算技术依赖影响
        
        Args:
            tech_sector: 技术领域
            tech_levels: 技术水平
            
        Returns:
            依赖影响值
        """
        dependencies = self.dependency_matrix.get(tech_sector, [])
        if not dependencies:
            return 0.0
        
        # 计算依赖技术的平均水平
        avg_dependency_level = 0.0
        valid_dependencies = 0
        
        for dep_tech in dependencies:
            if dep_tech in tech_levels:
                avg_dependency_level += tech_levels[dep_tech]
                valid_dependencies += 1
        
        if valid_dependencies == 0:
            return -0.1  # 无依赖技术，负面影响
        
        avg_dependency_level /= valid_dependencies
        
        # 计算依赖影响
        current_level = tech_levels.get(tech_sector, 0.0)
        dependency_impact = self.dependency_impact * (avg_dependency_level - current_level) * 0.01
        
        return dependency_impact
    
    def _is_country_sanctioned(self, country: str) -> bool:
        """
        检查国家是否受到制裁
        
        Args:
            country: 国家
            
        Returns:
            是否受到制裁
        """
        return country in self.sanction_targets
    
    def _check_for_breakthrough(self, country: str, tech_sector: str,
                               tech_levels: Dict[str, float],
                               random_state: np.random.RandomState) -> float:
        """
        检查是否发生技术突破
        
        Args:
            country: 国家
            tech_sector: 技术领域
            tech_levels: 技术水平
            random_state: 随机数生成器
            
        Returns:
            突破带来的技术提升
        """
        # 技术水平越高，越容易发生突破
        current_level = tech_levels.get(tech_sector, 0.0)
        breakthrough_chance = self.breakthrough_probability * (1.0 + current_level / self.max_tech_level)
        
        if random_state.random() < breakthrough_chance:
            # 发生技术突破
            breakthrough_size = random_state.uniform(5.0, 15.0)
            self.logger.info(f"技术突破: {country} 在 {tech_sector} 领域获得 {breakthrough_size:.2f} 点提升")
            return breakthrough_size
        
        return 0.0
    
    def _calculate_diffusion_impact(self, country: str, tech_sector: str,
                                   global_tech_state: Dict[str, Dict[str, float]],
                                   random_state: np.random.RandomState) -> float:
        """
        计算技术扩散影响
        
        Args:
            country: 国家
            tech_sector: 技术领域
            global_tech_state: 全球技术状态
            random_state: 随机数生成器
            
        Returns:
            扩散影响值
        """
        # 计算全球领先水平
        global_leading_level = 0.0
        for other_country, other_tech_levels in global_tech_state.items():
            if other_country != country and tech_sector in other_tech_levels:
                global_leading_level = max(global_leading_level, other_tech_levels[tech_sector])
        
        current_level = global_tech_state.get(country, {}).get(tech_sector, 0.0)
        
        # 如果该国家已经是领先者，没有扩散收益
        if current_level >= global_leading_level - 1.0:  # 允许小幅波动
            return 0.0
        
        # 计算扩散潜力
        diffusion_potential = global_leading_level - current_level
        
        # 随机扩散系数
        diffusion_factor = random_state.random() * self.diffusion_rate
        
        # 计算扩散影响
        diffusion_impact = diffusion_potential * diffusion_factor * 0.1
        
        return diffusion_impact
    
    def set_sanctions(self, country: str, tech_sectors: List[str]):
        """
        设置对特定国家的技术制裁
        
        Args:
            country: 国家
            tech_sectors: 制裁的技术领域列表
        """
        self.sanction_targets[country] = tech_sectors
        self.logger.info(f"设置制裁: {country} 的 {tech_sectors} 领域")
    
    def lift_sanctions(self, country: str):
        """
        解除对特定国家的制裁
        
        Args:
            country: 国家
        """
        if country in self.sanction_targets:
            del self.sanction_targets[country]
            self.logger.info(f"解除制裁: {country}")
    
    def get_global_tech_leader(self, tech_sector: str,
                             tech_state: Dict[str, Dict[str, float]]) -> Optional[str]:
        """
        获取特定技术领域的全球领先国家
        
        Args:
            tech_sector: 技术领域
            tech_state: 技术状态
            
        Returns:
            领先国家
        """
        max_level = -1
        leader = None
        
        for country, tech_levels in tech_state.items():
            if tech_sector in tech_levels and tech_levels[tech_sector] > max_level:
                max_level = tech_levels[tech_sector]
                leader = country
        
        return leader
    
    def calculate_tech_gap(self, country1: str, country2: str,
                          tech_sector: str,
                          tech_state: Dict[str, Dict[str, float]]) -> float:
        """
        计算两个国家在特定技术领域的差距
        
        Args:
            country1: 国家1
            country2: 国家2
            tech_sector: 技术领域
            tech_state: 技术状态
            
        Returns:
            技术差距 (country1 - country2)
        """
        level1 = tech_state.get(country1, {}).get(tech_sector, 0.0)
        level2 = tech_state.get(country2, {}).get(tech_sector, 0.0)
        
        return level1 - level2

# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 初始化技术模型
    tech_model = TechModel()
    
    # 创建初始技术状态
    initial_state = {
        "USA": {
            "artificial_intelligence": 80.0,
            "quantum_computing": 75.0,
            "biotechnology": 78.0,
            "semiconductors": 85.0
        },
        "China": {
            "artificial_intelligence": 75.0,
            "quantum_computing": 65.0,
            "renewable_energy": 82.0,
            "semiconductors": 60.0
        },
        "EU": {
            "biotechnology": 82.0,
            "renewable_energy": 78.0,
            "cybersecurity": 75.0
        }
    }
    
    # 设置制裁
    tech_model.set_sanctions("China", ["semiconductors", "quantum_computing"])
    
    # 演化技术状态
    random_state = np.random.RandomState(42)
    global_conditions = {
        "economic_health": {"USA": 0.95, "China": 0.9, "EU": 0.85},
        "political_stability": {"USA": 0.8, "China": 0.9, "EU": 0.85}
    }
    
    print("初始技术状态:")
    for country, tech in initial_state.items():
        print(f"{country}: {tech}")
    
    # 模拟10个时间步
    for step in range(10):
        new_state = tech_model.evolve(initial_state, global_conditions, random_state)
        initial_state = new_state
    
    print("\n演化后的技术状态:")
    for country, tech in initial_state.items():
        print(f"{country}: {tech}")
    
    # 获取全球领导者
    print("\n全球技术领导者:")
    for sector in ["artificial_intelligence", "quantum_computing", "semiconductors"]:
        leader = tech_model.get_global_tech_leader(sector, initial_state)
        print(f"{sector}: {leader}")
    
    # 计算技术差距
    print("\n技术差距 (USA - China):")
    for sector in ["artificial_intelligence", "semiconductors"]:
        gap = tech_model.calculate_tech_gap("USA", "China", sector, initial_state)
        print(f"{sector}: {gap:.2f}")