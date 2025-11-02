# Nexus 项目快速开始指南

本指南将帮助您快速设置和运行Nexus全球博弈模拟系统。

## 系统要求

- Python 3.8 或更高版本
- Docker 和 Docker Compose（推荐用于完整部署）
- 至少 8GB RAM（推荐 16GB 以上）
- 10GB 可用磁盘空间

## 快速部署选项

### 选项1：使用 Docker Compose（推荐）

1. **克隆代码库**

```bash
git clone https://github.com/your-org/Nexus.git
cd Nexus
```

2. **配置环境变量**

复制环境变量模板并根据需要修改：

```bash
cp .env.example .env
# 编辑 .env 文件，填入必要的API密钥
```

3. **启动所有服务**

使用 Makefile 快速启动：

```bash
make docker-compose-up
```

或直接使用 Docker Compose：

```bash
docker-compose up -d
```

4. **访问服务**

服务启动后，您可以访问以下地址：

- Nexus API: http://localhost:8000
- Grafana 仪表板: http://localhost:3000（用户名/密码: admin/admin）
- Prometheus: http://localhost:9090
- Jupyter Lab: http://localhost:8888（访问令牌在容器日志中）

### 选项2：本地安装

1. **克隆代码库**

```bash
git clone https://github.com/your-org/Nexus.git
cd Nexus
```

2. **创建虚拟环境**

```bash
make venv
# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
env\Scripts\activate      # Windows
```

3. **安装依赖**

```bash
make install
# 或安装开发依赖
make install-dev
```

4. **配置环境变量**

```bash
cp .env.example .env
# 编辑 .env 文件，填入必要的API密钥
```

5. **启动服务**

启动API服务器：

```bash
make server
```

或运行简单模拟：

```bash
make simulate
```

## 基本使用

### 运行模拟

1. **通过命令行运行模拟**

```bash
nexus-simulate --config config/config.yaml --output data/simulation_results
```

2. **通过API启动模拟**

```bash
curl -X POST http://localhost:8000/api/v1/simulations \
  -H "Content-Type: application/json" \
  -d '{"config_path": "config/config.yaml", "num_worldlines": 100, "time_horizon": 20}'
```

### 查看结果

模拟完成后，您可以：

1. **通过API查询结果**

```bash
curl http://localhost:8000/api/v1/simulations/{simulation_id}/results
```

2. **在Grafana仪表板中查看可视化结果**

访问 http://localhost:3000，导航到Nexus模拟仪表板。

3. **分析结果数据**

```bash
nexus-analyze --input data/simulation_results --output data/insights
```

## 首次运行配置

### 必要配置项

在`.env`文件中，以下配置项是必需的：

- `API_KEY_WORLD_BANK`：世界银行API密钥（可选，无密钥时使用有限访问）
- `API_KEY_NASA`：NASA API密钥（可选，无密钥时使用有限访问）
- `DATA_DIR`：数据存储目录
- `CACHE_DIR`：缓存目录
- `LOGS_DIR`：日志目录

### 模型参数配置

您可以通过编辑`config/model_params.yaml`文件来自定义模拟参数：

- 调整各领域模型权重
- 配置交互强度
- 设置随机种子以获得可重现的结果

## 常见问题

### 端口冲突

如果遇到端口冲突，请在`.env`文件中修改相关端口配置。

### API访问受限

如果第三方API访问受限，请确保您已配置正确的API密钥，或使用数据缓存功能。

### 性能问题

- 减少世界线数量（在配置文件中设置`num_worldlines`）
- 缩短时间跨度（设置`time_horizon`）
- 增加系统内存

## 下一步

- 查看[开发环境设置](development/development_setup.md)开始开发
- 阅读[系统架构](architecture/system_architecture.md)了解系统设计
- 参考[部署指南](deployment/deployment.md)进行生产环境部署

---

*文档最后更新时间：{current_date}*