"""
常量定义
同时通过预处理一些子树并整体作为常量以加快搜索进度
"""

import numpy as np
from operator import add, sub, mul, pow
from opt import *

# 定义运算符字典, 有数值错误会抛出到顶层
using_operators = {
    "1": add,
    "2": sub,
    "3": mul,
    "4": lambda x, y: x / y,
    "5": lambda x, y: pow(x, y),
    "6": lambda x: x
}

original_values = np.array([
    3.1415926535897932, # pi
    2.7182818284590452, # e
], dtype=np.float64)

# Pi/E, E/Pi, Pi+E, Pi-E, Pi E, E^Pi, Pi^E, Pi Pi, E E, Pi+Pi, E+E, Pi^Pi, E^E
sub_values = np.array([
    1.1557273497909217179,
    0.86525597943226508722,
    5.8598744820488384738,
    0.42331082513074800310,
    8.5397342226735670655,
    23.140692632779269006,
    22.459157718361045473, 
    9.8696044010893586188, 
    7.3890560989306502272, 
    6.2831853071795864769, 
    5.4365636569180904707, 
    36.462159607207911771,
    15.154262241479264190
], dtype=np.float64)

addition_values = np.array([
    114516.426867869998 # (pi pi pi)^pi/(pi-e)
], dtype=np.float64)

values = np.concatenate((original_values, addition_values, addition_values))
placeholder = [f"#{i+1}" for i in range(len(values))]