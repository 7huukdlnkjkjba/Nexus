# Phantom Crawler Nexus - 世界线模拟器

![Project Status](https://img.shields.io/badge/status-active-green)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

# 你相信命中注定吗？你相信穿越吗？

## 项目概述
Phantom Crawler Nexus 是一个先进的世界线模拟系统，旨在探索和预测全球系统在不同条件下的可能演化路径。其核心机制是持续地"孵化"、"筛选"和"进化"虚拟世界，最终保留最可能发生的未来，帮助理解塑造现实的核心驱动力。

## 核心理念

### 1. 多领域整合模拟
基于经济学、政治学、技术发展、气候变化和危机响应等多个领域的相互作用模型，创建全面的全球系统模拟。

### 2. 世界线分叉与演化
在关键决策点创建多条平行世界线，并通过各领域模型的协同作用追踪其演化轨迹。

### 3. 智能评估与筛选
使用先进的评估算法计算各世界线的生存概率和价值，持续优化世界线池，保留最有价值的演化路径。

## 架构设计

### 核心组件
- **Timeline Manager**: 管理多条世界线的创建、演化和修剪
- **Simulation Coordinator**: 协调各子模型并管理整体模拟流程
- **Event System**: 生成和处理各类全球事件
- **WorldLine Evaluator**: 评估世界线的生存概率和价值

### 领域模型
- **Economic Model**: 模拟全球经济系统演化
- **Political Model**: 模拟全球政治格局变化
- **Technology Model**: 模拟技术发展和创新进程
- **Climate Model**: 模拟气候变化和环境因素
- **Crisis Response Model**: 模拟对全球危机的响应机制

### 核心特性

- **多领域整合模拟**: 融合经济、政治、技术、气候和危机响应等多个领域的相互作用
- **世界线分叉与演化**: 基于关键决策点创建多条平行世界线并追踪其演化
- **先进的事件系统**: 模拟各类全球事件（经济危机、技术突破、冲突、疫情等）
- **智能评估机制**: 评估各世界线的生存概率和价值
- **灵活的配置系统**: 支持高度自定义的模拟参数和模型设置
- **详细的结果分析**: 生成全面的报告和数据可视化

## 技术栈
- **编程语言**: Python 3.8+
- **核心依赖**: NumPy, Pandas
- **并行计算**: Python 多线程/多进程支持
- **数据存储**: JSON 格式文件存储
- **日志记录**: Python 标准日志库
- **配置管理**: JSON 配置文件

## 安装与配置

### 系统要求

- Python 3.8 或更高版本
- 推荐配置：8GB+ RAM，多核处理器
- 存储空间：至少 1GB 用于安装和模拟结果

### 安装步骤

1. **克隆项目仓库**

```bash
git clone https://your-repository-url/phantom-crawler-nexus.git
cd phantom-crawler-nexus
```

2. **创建虚拟环境**（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **验证安装**

```bash
python test_simulation.py
```

## 使用指南

### 运行标准模拟

```bash
python main.py
```

### 使用自定义配置

1. 创建配置文件 `custom_config.json`

```json
{
  "simulation": {
    "initial_year": 2023,
    "target_year": 2100,
    "max_timelines": 20
  },
  "models": {
    "economic": {
      "initial_gdp": 100.0,
      "gdp_growth_volatility": 0.02
    }
    // 其他配置...
  }
}
```

2. 使用配置文件运行

```bash
python main.py --config custom_config.json
```

### 命令行参数

- `--config <path>` - 指定配置文件路径
- `--seed <number>` - 设置随机种子（用于可重复结果）
- `--initial-year <year>` - 设置初始年份
- `--target-year <year>` - 设置目标年份
- `--output <path>` - 指定输出目录路径
- `--debug` - 启用调试模式

### 运行测试

执行小规模测试以验证系统功能：

```bash
python test_simulation.py
```

## 输出结果

模拟完成后，结果将保存在输出目录中，包括：

- `timelines/` - 各世界线的详细数据
- `events/` - 生成的所有事件记录
- `metrics/` - 评估指标和统计数据
- `config.json` - 使用的配置参数
- `simulation_summary.md` - 模拟结果摘要报告

## 项目结构

```
PhantomCrawler/Nexus/
├── main.py                # 主程序入口
├── test_simulation.py     # 测试脚本
├── requirements.txt       # 依赖列表
├── README.md              # 项目文档
├── src/
│   ├── core/              # 核心组件
│   │   ├── simulation_coordinator.py  # 模拟协调器
│   │   ├── timeline_manager.py        # 时间线管理器
│   │   └── event_system.py            # 事件系统
│   ├── data/              # 数据相关
│   │   ├── data_types.py  # 数据类型定义
│   │   └── data_processor.py  # 数据处理器
│   └── models/            # 领域模型
│       ├── evaluator.py            # 世界线评估器
│       ├── economic_model.py       # 经济模型
│       ├── political_model.py      # 政治模型
│       ├── technology_model.py     # 技术模型
│       ├── climate_model.py        # 气候模型
│       └── crisis_response_model.py  # 危机响应模型
```

## 高级用法

### 自定义模型

您可以通过扩展现有模型类来实现自定义行为：

```python
from src.models.economic_model import EconomicModel

class CustomEconomicModel(EconomicModel):
    def __init__(self, config):
        super().__init__(config)
        # 添加自定义初始化
    
    def evolve_economy(self, current_state, events):
        # 实现自定义演化逻辑
        # ...
        return new_state
```

### 集成新数据源

系统支持集成外部数据源以增强模拟准确性：

1. 在 `src/data` 中创建新的数据收集器
2. 更新数据处理逻辑以整合新数据
3. 修改相应模型以利用新数据源

## 常见问题

### Q: 模拟运行太慢怎么办？

A: 您可以：
- 减少最大世界线数量
- 增加修剪阈值
- 减少每年决策点数量
- 在测试阶段使用较小的时间范围

### Q: 如何确保结果可重现？

A: 使用固定的随机种子运行：

```bash
python main.py --seed 42
```

### Q: 如何解释世界线的价值分数？

A: 价值分数基于多个因素计算，包括：
- 经济发展水平
- 政治稳定性
- 技术创新程度
- 环境可持续性
- 社会福利指标
```

## 开发指南

### 添加新功能

1. 创建功能分支
2. 实现新功能或修复
3. 编写测试用例
4. 提交 Pull Request

### 代码风格

项目遵循 PEP 8 代码风格规范。请确保：
- 使用描述性变量名和函数名
- 添加适当的文档字符串
- 保持代码简洁和可读性

### 贡献指南
1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 许可证
本项目采用 MIT 许可证 - 详情请参阅 LICENSE 文件

## 联系方式

- 项目主页: [Phantom Crawler Nexus](https://your-project-url)
- 问题反馈: [GitHub Issues](https://your-repository-url/issues)
- 维护者: [Your Name/Team] - contact@example.com

## 免责声明
Nexus引擎仅用于学术研究和战略规划目的。模拟结果不应被视为绝对预测，而应作为决策支持工具。使用本系统进行的任何分析都应结合其他信息源和专业判断。
---

*"探索平行世界的可能性，模拟全球系统的演化路径"*
