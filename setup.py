from setuptools import setup, find_packages

setup(
    name="tinyinfra",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "accelerate",
        "click",
        "pyyaml",
        "rich",  # For beautiful CLI output
        "numpy",
        "torchvision",
        "scipy",
        "tabulate",
        "vllm",
        "setuptools"
    ],
    entry_points={
        'console_scripts': [
            'tinyinfra=tinyinfra.cli:cli',
        ],
    },
    author="Ruihan",
    description="LLM Tiny Infra",
    python_requires='>=3.12',
)
