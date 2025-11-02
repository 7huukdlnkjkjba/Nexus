#!/usr/bin/env python3
"""
Nexus - 全球博弈模拟与策略超前推演引擎
安装配置文件
"""

from setuptools import setup, find_packages
import os
import sys

# 检查Python版本
if sys.version_info < (3, 9):
    raise RuntimeError("Nexus 要求 Python 3.9 或更高版本")

# 读取README.md作为长描述
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Nexus - 全球博弈模拟与策略超前推演引擎"

# 读取依赖列表
def parse_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        requirements = []
        for line in f:
            line = line.strip()
            # 忽略注释和空行
            if line and not line.startswith("#") and not line.startswith("-"):
                requirements.append(line)
    return requirements

try:
    install_requires = parse_requirements("requirements.txt")
except FileNotFoundError:
    # 默认依赖列表
    install_requires = [
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "requests>=2.27.0",
        "python-dotenv>=0.20.0",
        "tqdm>=4.62.0",
        "psutil>=5.9.0",
        "pytest>=7.0.0",
        "coverage>=6.3.0",
        "black>=22.3.0",
        "isort>=5.10.0",
        "flake8>=4.0.0"
    ]

# 额外依赖选项
extra_requires = {
    "dev": [
        "pytest-cov>=3.0.0",
        "pre-commit>=2.17.0",
        "sphinx>=4.4.0",
        "sphinx-rtd-theme>=1.0.0",
        "mypy>=0.942",
    ],
    "ml": [
        "tensorflow>=2.8.0",
        "keras>=2.8.0",
        "torch>=1.10.0",
        "prophet>=1.0.1",
    ],
    "visualization": [
        "plotly>=5.6.0",
        "dash>=2.3.0",
        "seaborn>=0.11.2",
    ],
    "api": [
        "fastapi>=0.75.0",
        "uvicorn>=0.17.0",
        "pydantic>=1.9.0",
    ],
    "distributed": [
        "dask>=2022.03.0",
        "ray>=1.12.0",
    ],
    "database": [
        "sqlalchemy>=1.4.0",
        "psycopg2-binary>=2.9.0",
        "redis>=4.3.0",
    ],
    "streaming": [
        "kafka-python>=2.0.2",
        "confluent-kafka>=1.8.0",
    ],
}

# 合并所有额外依赖
all_extra = []
for extras in extra_requires.values():
    all_extra.extend(extras)
extra_requires["all"] = list(set(all_extra))

# 项目元数据
setup(
    name="nexus-simulation-engine",
    version="1.0.0",
    description="全球博弈模拟与策略超前推演引擎",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nexus Development Team",
    author_email="team@nexus-simulator.com",  # 示例邮箱
    url="https://github.com/nexus-simulation/nexus",  # 示例URL
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Simulation",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Social Sciences :: Sociology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "simulation",
        "strategy",
        "game-theory",
        "foresight",
        "global",
        "complex-systems",
        "artificial-intelligence",
        "climate-change",
        "economic-modeling",
        "political-science",
        "technological-forecasting",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extra_requires,
    include_package_data=True,
    package_data={
        "": [
            "config/*.yaml",
            "data/*.json",
            "data/*.csv",
            "*.md",
            "*.txt",
        ],
    },
    data_files=[
        ("config", ["config/config.yaml", "config/model_params.yaml", "config/data_sources.yaml"]),
        (".", [".env.example"]),
    ],
    entry_points={
        "console_scripts": [
            "nexus=nexus.cli:main",
            "nexus-simulate=nexus.cli:simulate",
            "nexus-validate=nexus.cli:validate",
            "nexus-analyze=nexus.cli:analyze",
            "nexus-visualize=nexus.cli:visualize",
            "nexus-server=nexus.cli:server",
        ],
    },
    zip_safe=False,
    project_urls={
        "Documentation": "https://nexus-simulation.readthedocs.io/",
        "Source": "https://github.com/nexus-simulation/nexus",
        "Tracker": "https://github.com/nexus-simulation/nexus/issues",
    },
    # 测试配置
    test_suite="tests",
    tests_require=["pytest", "pytest-cov"],
    # 安装后操作
    cmdclass={},
)

# 打印安装信息
if __name__ == "__main__":
    print("Nexus 安装配置加载完成")
    print(f"Python版本要求: >= 3.9")
    print(f"当前Python版本: {sys.version}")
    print(f"安装依赖数: {len(install_requires)}")
    print(f"可选功能模块: {list(extra_requires.keys())}")