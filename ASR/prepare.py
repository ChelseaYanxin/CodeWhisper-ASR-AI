# load_tedlium.py
from datasets import load_dataset

def load_tedlium_dataset():
    """加载 TED-LIUM 数据集并返回"""
    tedlium = load_dataset("LIUM/tedlium", "release1")
    return tedlium