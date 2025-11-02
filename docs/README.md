# Nexus 项目文档

## 目录结构

```
docs/
├── README.md                  # 文档目录说明
├── architecture/              # 架构设计文档
├── api/                       # API文档（自动生成）
├── deployment/                # 部署指南
├── development/               # 开发指南
├── models/                    # 模型说明文档
├── troubleshooting/           # 故障排除指南
└── quickstart.md              # 快速开始指南
```

## 文档索引

### 快速开始
- [快速开始指南](quickstart.md) - 快速部署和运行Nexus项目

### 架构设计
- [系统架构](architecture/system_architecture.md) - Nexus系统整体架构
- [数据流](architecture/data_flow.md) - 系统数据流设计
- [组件说明](architecture/components.md) - 核心组件详细说明

### 开发文档
- [开发环境设置](development/development_setup.md) - 开发环境配置指南
- [编码规范](development/coding_standards.md) - 项目编码规范
- [测试指南](development/testing_guide.md) - 测试编写和运行指南
- [贡献流程](development/contributing.md) - 代码贡献流程

### 部署文档
- [本地部署](deployment/local_deployment.md) - 本地环境部署指南
- [Docker部署](deployment/docker_deployment.md) - Docker容器化部署
- [生产环境部署](deployment/production_deployment.md) - 生产环境部署最佳实践
- [配置说明](deployment/configuration.md) - 配置文件详解

### 模型文档
- [领域模型](models/domain_models.md) - 经济、政治、技术、气候等领域模型
- [交互模型](models/interaction_model.md) - 领域间交互机制
- [洞察提取](models/insight_extraction.md) - 洞察提取算法
- [模拟引擎](models/simulation_engine.md) - 世界线模拟引擎

### 故障排除
- [常见问题](troubleshooting/faq.md) - 常见问题解答
- [错误代码](troubleshooting/error_codes.md) - 错误代码和解决方案
- [性能优化](troubleshooting/performance_optimization.md) - 性能调优建议

## 文档维护

### 生成API文档

使用Sphinx自动生成API文档：

```bash
make docs
```

### 更新文档

请在更新代码的同时更新相关文档，确保文档与代码保持同步。所有文档应使用Markdown格式编写，并遵循项目文档风格指南。

### 文档贡献

1. 在对应模块的文档文件中添加或修改内容
2. 确保文档准确、清晰且易于理解
3. 代码示例应保持最新并可运行
4. 提交前预览文档渲染效果

## 文档预览

本地预览Markdown文档，可以使用以下工具：

- VS Code + Markdown扩展
- Typora
- MarkText
- GitHub/GitLab预览功能

## 文档版本控制

文档版本应与代码版本保持一致。在发布新版本时，请确保更新相关文档内容。

---

*文档最后更新时间：{current_date}*