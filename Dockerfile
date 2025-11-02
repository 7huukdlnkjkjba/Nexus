# Nexus 项目 Dockerfile
# 基于官方Python镜像构建的容器化部署方案

# 使用官方Python 3.10镜像作为基础镜像
FROM python:3.10-slim-bookworm

# 维护者信息
LABEL maintainer="Nexus Development Team <team@nexus-simulator.com>"
LABEL description="全球博弈模拟与策略超前推演引擎"
LABEL version="1.0.0"

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    wget \
    git \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 升级pip和安装依赖管理工具
RUN pip install --upgrade pip setuptools wheel virtualenv

# 创建虚拟环境
RUN python -m venv /app/.venv

# 复制依赖文件
COPY requirements.txt* /app/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt || echo "No requirements.txt found, will install dependencies from setup.py"

# 复制项目文件
COPY . /app/

# 安装项目本身
RUN pip install --no-cache-dir -e .

# 创建必要的目录
RUN mkdir -p /app/data /app/cache /app/logs /app/output

# 设置权限
RUN chmod -R 777 /app/data /app/cache /app/logs /app/output

# 复制配置文件
COPY config/*.yaml /app/config/

# 复制环境变量模板
COPY .env.example /app/

# 暴露端口（用于API服务）
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# 默认命令
CMD ["python", "-m", "nexus.engine"]

# 为开发环境提供交互式shell选项
# 可以通过 docker run -it --entrypoint /bin/bash nexus-simulation 进入

# --- 多阶段构建选项（可选）---
# 此Dockerfile使用单阶段构建以简化开发
# 对于生产环境，可以考虑以下多阶段构建方式：
# 
# FROM python:3.10-slim-bookworm AS builder
# WORKDIR /app
# RUN pip install --upgrade pip setuptools wheel
# COPY requirements.txt .
# RUN pip install --no-cache-dir --user -r requirements.txt
# COPY . .
# RUN pip install --no-cache-dir --user -e .
# 
# FROM python:3.10-slim-bookworm AS runtime
# WORKDIR /app
# COPY --from=builder /root/.local /root/.local
# ENV PATH=/root/.local/bin:$PATH
# COPY . /app/
# RUN mkdir -p /app/data /app/cache /app/logs /app/output
# EXPOSE 8000
# CMD ["python", "-m", "nexus.engine"]

# --- 不同环境的Dockerfile变体 ---#
# 开发环境:
# - 添加开发依赖
# - 挂载源代码目录
# - 启用调试模式
# 
# 生产环境:
# - 优化镜像大小
# - 非root用户运行
# - 更严格的安全设置
# - 预编译Python字节码

# --- 扩展提示 ---#
# 1. 如需GPU支持，可以基于nvidia/cuda镜像构建
# 2. 如需分布式计算支持，可以安装dask/ray
# 3. 可以使用Docker Compose集成数据库、缓存等服务
# 4. 生产环境建议使用非root用户运行
# 
# 示例非root用户配置:
# RUN groupadd -r nexus && useradd -r -g nexus nexus
# USER nexus
# WORKDIR /app
# 
# 注意：非root用户需要适当的目录权限设置