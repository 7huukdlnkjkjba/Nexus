# Nexus 项目 Makefile
# 封装常用命令，简化开发和部署流程

# 默认目标
.DEFAULT_GOAL := help

# 变量定义
PYTHON := python3
PIP := pip
VENV := .venv
PROJECT_DIR := $(shell pwd)
SRC_DIR := $(PROJECT_DIR)/src
TESTS_DIR := $(PROJECT_DIR)/tests
CONFIG_DIR := $(PROJECT_DIR)/config
LOGS_DIR := $(PROJECT_DIR)/logs
DATA_DIR := $(PROJECT_DIR)/data
CACHE_DIR := $(PROJECT_DIR)/cache
OUTPUT_DIR := $(PROJECT_DIR)/output

# 颜色定义
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# 检查虚拟环境
venv-exists:
	@if [ ! -d "$(VENV)" ]; then \
	    echo -e "$(RED)虚拟环境不存在，请先运行 'make venv'$(NC)"; \
	    exit 1; \
	fi

# 帮助信息
help:
	@echo -e "$(BLUE)Nexus 项目命令行工具$(NC)"
	@echo -e "======================"
	@echo -e "$(GREEN)开发相关命令:$(NC)"
	@echo -e "  make venv              创建虚拟环境"
	@echo -e "  make install           安装项目依赖"
	@echo -e "  make install-dev       安装开发依赖"
	@echo -e "  make install-all       安装所有可选依赖"
	@echo -e "  make clean             清理构建文件"
	@echo -e "  make lint              运行代码检查"
	@echo -e "  make format            格式化代码"
	@echo -e "  make test              运行单元测试"
	@echo -e "  make test-cov          运行测试并生成覆盖率报告"
	@echo -e "  make docs              生成API文档"
	@echo -e "\n$(GREEN)运行相关命令:$(NC)"
	@echo -e "  make run               运行主程序"
	@echo -e "  make simulate          运行模拟"
	@echo -e "  make analyze           分析结果"
	@echo -e "  make validate          验证配置"
	@echo -e "  make server            启动API服务器"
	@echo -e "\n$(GREEN)Docker相关命令:$(NC)"
	@echo -e "  make docker-build      构建Docker镜像"
	@echo -e "  make docker-run        运行Docker容器"
	@echo -e "  make docker-compose-up 启动所有服务（开发环境）"
	@echo -e "  make docker-compose-prod 启动所有服务（生产环境）"
	@echo -e "  make docker-compose-down 停止所有服务"
	@echo -e "  make docker-logs       查看容器日志"
	@echo -e "\n$(GREEN)维护相关命令:$(NC)"
	@echo -e "  make create-dirs       创建必要的目录结构"
	@echo -e "  make update-deps       更新依赖版本"
	@echo -e "  make check-updates     检查依赖更新"
	@echo -e "  make backup-config     备份配置文件"
	@echo -e "  make clean-cache       清理缓存文件"
	@echo -e "  make clean-logs        清理日志文件"
	@echo -e "  make version           显示项目版本"

# 创建虚拟环境
venv:
	@echo -e "$(GREEN)创建虚拟环境...$(NC)"
	@if [ -d "$(VENV)" ]; then \
	    echo -e "$(YELLOW)虚拟环境已存在，将其删除...$(NC)"; \
	    rm -rf $(VENV); \
	fi
	@$(PYTHON) -m venv $(VENV)
	@echo -e "$(GREEN)虚拟环境创建成功: $(VENV)$(NC)"
	@echo -e "$(YELLOW)请运行 'source $(VENV)/bin/activate' 激活虚拟环境$(NC)"

# 安装项目依赖
install: venv-exists
	@echo -e "$(GREEN)安装项目依赖...$(NC)"
	@$(VENV)/bin/$(PIP) install --upgrade pip setuptools wheel
	@if [ -f "requirements.txt" ]; then \
	    $(VENV)/bin/$(PIP) install --no-cache-dir -r requirements.txt; \
	fi
	@$(VENV)/bin/$(PIP) install --no-cache-dir -e .
	@echo -e "$(GREEN)依赖安装完成$(NC)"

# 安装开发依赖
install-dev: venv-exists
	@echo -e "$(GREEN)安装开发依赖...$(NC)"
	@$(VENV)/bin/$(PIP) install --no-cache-dir -e .[dev]
	@echo -e "$(GREEN)开发依赖安装完成$(NC)"

# 安装所有可选依赖
install-all: venv-exists
	@echo -e "$(GREEN)安装所有可选依赖...$(NC)"
	@$(VENV)/bin/$(PIP) install --no-cache-dir -e .[all]
	@echo -e "$(GREEN)所有依赖安装完成$(NC)"

# 清理构建文件
clean:
	@echo -e "$(GREEN)清理构建文件...$(NC)"
	@rm -rf $(PROJECT_DIR)/build
	@rm -rf $(PROJECT_DIR)/dist
	@rm -rf $(PROJECT_DIR)/*.egg-info
	@find $(SRC_DIR) -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find $(TESTS_DIR) -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find $(PROJECT_DIR) -name "*.pyc" -delete
	@find $(PROJECT_DIR) -name "*.pyo" -delete
	@echo -e "$(GREEN)清理完成$(NC)"

# 运行代码检查
lint: venv-exists
	@echo -e "$(GREEN)运行代码检查...$(NC)"
	@$(VENV)/bin/flake8 $(SRC_DIR)
	@$(VENV)/bin/mypy $(SRC_DIR)
	@$(VENV)/bin/isort --check-only --diff $(SRC_DIR)
	@echo -e "$(GREEN)代码检查完成$(NC)"

# 格式化代码
format: venv-exists
	@echo -e "$(GREEN)格式化代码...$(NC)"
	@$(VENV)/bin/black $(SRC_DIR)
	@$(VENV)/bin/isort $(SRC_DIR)
	@echo -e "$(GREEN)代码格式化完成$(NC)"

# 运行单元测试
test: venv-exists
	@echo -e "$(GREEN)运行单元测试...$(NC)"
	@$(VENV)/bin/pytest $(TESTS_DIR) -v

# 运行测试并生成覆盖率报告
test-cov: venv-exists
	@echo -e "$(GREEN)运行测试并生成覆盖率报告...$(NC)"
	@$(VENV)/bin/pytest $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=term --cov-report=html
	@echo -e "$(GREEN)覆盖率报告生成在 htmlcov/ 目录中$(NC)"

# 生成API文档
docs:
	@echo -e "$(GREEN)生成API文档...$(NC)"
	@mkdir -p $(PROJECT_DIR)/docs/api
	@$(VENV)/bin/sphinx-apidoc -o $(PROJECT_DIR)/docs/api $(SRC_DIR)
	@echo -e "$(GREEN)API文档生成完成$(NC)"

# 运行主程序
run: venv-exists
	@echo -e "$(GREEN)运行Nexus主程序...$(NC)"
	@$(VENV)/bin/python -m nexus.engine

# 运行模拟
simulate: venv-exists
	@echo -e "$(GREEN)运行模拟...$(NC)"
	@$(VENV)/bin/nexus-simulate

# 分析结果
analyze: venv-exists
	@echo -e "$(GREEN)分析模拟结果...$(NC)"
	@$(VENV)/bin/nexus-analyze

# 验证配置
validate: venv-exists
	@echo -e "$(GREEN)验证配置文件...$(NC)"
	@$(VENV)/bin/nexus-validate

# 启动API服务器
server: venv-exists
	@echo -e "$(GREEN)启动API服务器...$(NC)"
	@$(VENV)/bin/nexus-server

# 构建Docker镜像
docker-build:
	@echo -e "$(GREEN)构建Docker镜像...$(NC)"
	@docker build -t nexus-simulation:latest .
	@echo -e "$(GREEN)Docker镜像构建完成: nexus-simulation:latest$(NC)"

# 运行Docker容器
docker-run:
	@echo -e "$(GREEN)运行Docker容器...$(NC)"
	@docker run --rm -it \
	    -p 8000:8000 \
	    -v $(PWD)/data:/app/data \
	    -v $(PWD)/cache:/app/cache \
	    -v $(PWD)/logs:/app/logs \
	    -v $(PWD)/output:/app/output \
	    -v $(PWD)/config:/app/config \
	    nexus-simulation:latest

# 启动所有服务（开发环境）
docker-compose-up:
	@echo -e "$(GREEN)启动所有服务（开发环境）...$(NC)"
	@docker-compose up -d
	@echo -e "$(GREEN)服务启动完成，请访问:$(NC)"
	@echo -e "  Nexus API: http://localhost:8000"
	@echo -e "  Grafana: http://localhost:3000"
	@echo -e "  Prometheus: http://localhost:9090"
	@echo -e "  Jupyter Lab: http://localhost:8888"

# 启动所有服务（生产环境）
docker-compose-prod:
	@echo -e "$(GREEN)启动所有服务（生产环境）...$(NC)"
	@docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 停止所有服务
docker-compose-down:
	@echo -e "$(GREEN)停止所有服务...$(NC)"
	@docker-compose down

# 查看容器日志
docker-logs:
	@echo -e "$(GREEN)查看容器日志...$(NC)"
	@docker-compose logs -f

# 创建必要的目录结构
create-dirs:
	@echo -e "$(GREEN)创建必要的目录结构...$(NC)"
	@mkdir -p $(LOGS_DIR)
	@mkdir -p $(DATA_DIR)
	@mkdir -p $(CACHE_DIR)
	@mkdir -p $(OUTPUT_DIR)
	@mkdir -p $(TESTS_DIR)/data
	@mkdir -p $(TESTS_DIR)/output
	@mkdir -p $(PROJECT_DIR)/docs/api
	@mkdir -p $(PROJECT_DIR)/notebooks
	@echo -e "$(GREEN)目录创建完成$(NC)"

# 更新依赖版本
update-deps:
	@echo -e "$(GREEN)更新依赖版本...$(NC)"
	@$(VENV)/bin/$(PIP) install --upgrade -r requirements.txt
	@echo -e "$(GREEN)依赖更新完成$(NC)"

# 检查依赖更新
check-updates:
	@echo -e "$(GREEN)检查依赖更新...$(NC)"
	@$(VENV)/bin/$(PIP) list --outdated

# 备份配置文件
backup-config:
	@echo -e "$(GREEN)备份配置文件...$(NC)"
	@BACKUP_DIR=$(PROJECT_DIR)/backups/$(shell date +%Y%m%d_%H%M%S)
	@mkdir -p $$BACKUP_DIR
	@cp -r $(CONFIG_DIR)/* $$BACKUP_DIR/
	@cp $(PROJECT_DIR)/.env* $$BACKUP_DIR/ 2>/dev/null || true
	@echo -e "$(GREEN)配置备份完成: $$BACKUP_DIR$(NC)"

# 清理缓存文件
clean-cache:
	@echo -e "$(GREEN)清理缓存文件...$(NC)"
	@rm -rf $(CACHE_DIR)/* 2>/dev/null || true
	@echo -e "$(GREEN)缓存清理完成$(NC)"

# 清理日志文件
clean-logs:
	@echo -e "$(GREEN)清理日志文件...$(NC)"
	@find $(LOGS_DIR) -name "*.log*" -delete 2>/dev/null || true
	@echo -e "$(GREEN)日志清理完成$(NC)"

# 显示项目版本
version: venv-exists
	@echo -e "$(GREEN)Nexus 项目版本信息:$(NC)"
	@$(VENV)/bin/python -c "import nexus; print(f'版本: {nexus.__version__}')" 2>/dev/null || echo "无法获取版本信息"

# 检查系统环境
check-env:
	@echo -e "$(GREEN)检查系统环境...$(NC)"
	@$(PYTHON) --version
	@echo "操作系统: $(shell uname -a)"
	@echo "Python路径: $(PYTHON)"
	@echo "虚拟环境: $(VENV)"
	@if [ -d "$(VENV)" ]; then \
	    echo -e "$(GREEN)虚拟环境已安装$(NC)"; \
	    $(VENV)/bin/$(PIP) --version; \
	else \
	    echo -e "$(RED)虚拟环境未安装$(NC)"; \
	fi

# 安装预提交钩子（开发用）
install-pre-commit: venv-exists
	@echo -e "$(GREEN)安装预提交钩子...$(NC)"
	@$(VENV)/bin/$(PIP) install pre-commit
	@$(VENV)/bin/pre-commit install

# 运行预提交检查
pre-commit: venv-exists
	@echo -e "$(GREEN)运行预提交检查...$(NC)"
	@$(VENV)/bin/pre-commit run --all-files

# 导出依赖列表
export-deps:
	@echo -e "$(GREEN)导出当前依赖列表...$(NC)"
	@$(VENV)/bin/$(PIP) freeze > requirements.lock.txt
	@echo -e "$(GREEN)依赖已导出到 requirements.lock.txt$(NC)"

# 启动开发服务器（带热重载）
start-dev-server: venv-exists
	@echo -e "$(GREEN)启动开发服务器...$(NC)"
	@$(VENV)/bin/uvicorn nexus.api.server:app --reload --host 0.0.0.0 --port 8000

# 运行性能测试
benchmark: venv-exists
	@echo -e "$(GREEN)运行性能测试...$(NC)"
	@$(VENV)/bin/python -m pytest $(TESTS_DIR)/benchmarks -v

.PHONY: help venv install install-dev install-all clean lint format test test-cov docs \
        run simulate analyze validate server docker-build docker-run \
        docker-compose-up docker-compose-prod docker-compose-down docker-logs \
        create-dirs update-deps check-updates backup-config clean-cache clean-logs \
        version check-env install-pre-commit pre-commit export-deps start-dev-server benchmark