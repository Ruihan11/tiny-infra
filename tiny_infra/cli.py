import click
from pathlib import Path
import yaml
from rich.console import Console
from rich.table import Table

console = Console()

@click.group()
@click.version_option(version='0.1.0')
def cli():
    """
    TinyInfra - LLM Inference Optimization Toolkit
    
    Benchmark and optimize large language models for production deployment.
    """
    pass


@cli.group()
def benchmark():
    """Run performance benchmarks"""
    pass