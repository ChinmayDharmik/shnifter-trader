"""
Setup script for Shnifter Trader
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="shnifter-trader",
    version="1.0.0",
    author="Shnifter Trader Team",
    author_email="contact@shnifter.com",
    description="Advanced AI-Powered Trading Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shnifter/trader",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=2.1.0",
        "pandas>=2.2.3",
        "scikit-learn>=1.6",
        "PySide6>=6.8.0",
        "yfinance>=0.2.0",
        "alpha-vantage>=2.3.1",
        "requests>=2.31.0",
        "pydantic>=2.0.0",
        "nltk>=3.8.1",
        "prophet>=1.1.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "openai>=1.93.0",
        "plotly>=6.2.0",
        "autogen-agentchat>=0.4.0",
        "chromadb>=0.5.0",
        "langgraph>=0.2.0",
        "streamlit>=1.40.0",
        "dash>=2.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "poetry>=1.6.0",
        ],
        "trading": [
            "backtrader>=1.9.78",
            "ccxt>=4.0.0",
            "python-binance>=1.0.19",
            "ta>=0.10.2",
            "pandas-ta>=0.3.14b0",
        ],
        "ai": [
            "transformers>=4.35.0",
            "torch>=2.0.0",
            "anthropic>=0.18.0",
            "langchain>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "shnifter-trader=Multi_Model_Trading_Bot:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.toml", "*.cfg", "*.ini"],
    },
)
