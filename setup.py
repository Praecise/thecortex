from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tenzro-cortex",
    version="0.1.0",
    author="Tenzro Network",
    author_email="dev@tenzro.network",
    description="Core AI model for the Tenzro Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tenzro/cortex",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "websockets>=10.0.0",
        "asyncio>=3.4.3",
        "pydantic>=1.9.0",
        "msgpack>=1.0.3",
        "tensorboard>=2.8.0",
        "cryptography>=3.4.7",  # For encryption
        "pyyaml>=5.4.1",       # For configuration
        "prometheus-client>=0.12.0",  # For metrics
        "structlog>=21.1.0",   # For structured logging
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "psutil>=5.8.0",    # For performance monitoring
            "gputil>=1.4.0"     # For GPU monitoring
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    }
)