"""
常量定义
同时通过预处理一些子树并整体作为常量以加快搜索进度
"""

import multiprocessing as mp
from operator import add, sub, mul, pow
from opt import *
from wl import consts_pool
from schedule import wolf

# 运算符字典
using_operators = {
    "1": add,
    "2": sub,
    "3": mul,
    "4": lambda x, y: x / y,
    "5": pow,
    "6": lambda x: x
}

# 定义参与运算的常数，如果使用了 Wolfram 引擎可以更简明地定义
# 在wl.py中设置常量
# wolfram仅在主进程加载，附属进程通过传参获得values

if opt.enable_wolfram and mp.current_process().name == "MainProcess":
    values_wolf = wolf.wolfram_evaluate("N[{" + consts_pool + "},17]")
    values = [float(i) for i in list(values_wolf)]

# 兼容非wolfram
else:
    values = [
        3.1415926535897932, 2.7182818284590452, 1.1557273497909217179,
        0.86525597943226508722, 5.8598744820488384738, 0.42331082513074800310,
        8.5397342226735670655, 23.140692632779269006, 22.459157718361045473, 
        9.8696044010893586188,  7.3890560989306502272,  6.2831853071795864769, 
        5.4365636569180904707,  36.462159607207911771, 15.154262241479264190,
        114516.426867869998
    ]

# 变量占位符，形式为 #k
placeholder = [f"#{i+1}" for i in range(len(values))]

# cost映射
cost_dict = {
    "Pi": 1,
    "E": 1,
    "Plus": 3,
    "Times": 12,
    "Power": 72
}