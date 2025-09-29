from setuptools import setup, find_packages

setup(
    name="tinyinfra",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.36.0",
        "click>=8.1.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",  # For beautiful CLI output
        "numpy>=1.24.0",
    ],
    entry_points={
        'console_scripts': [
            'tinyinfra=tinyinfra.cli:cli',
        ],
    },
    author="Ruihan",
    description="LLM Tiny Infra",
    python_requires='>=3.8',
)
