"""
统一日志配置
"""
import logging
import sys


def setup_logging(level: str = "INFO"):
    """配置全局日志格式和级别"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
