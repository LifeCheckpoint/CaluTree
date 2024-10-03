import numpy as np
from operator import add, sub, mul, pow
from tai.opt import opt
from tai.wl import *
from schedule import wolf


# 运算符字典，注意，为了taichi正常编译，运算符字典已经写死，请勿随意更改
operators_symbol_dict = {
    "+": add,
    "-": sub,
    "*": mul,
    "/": lambda x, y: x / y,
    "^": pow,
    "I": lambda x: x
}
operators_symbol = np.array(list(operators_symbol_dict.keys()))
operators_symbol_num = len(operators_symbol)


# 定义参与运算的常数，如果使用了 Wolfram 引擎可以更简明地定义
# 在wl.py中设置常量
if opt.enable_wolfram:
    values_wolf = wolf.wolfram_evaluate("N[{" + consts_pool + "},17]")
    values = np.array([float(i) for i in list(values_wolf)])
# 兼容非wolfram
else:
    values = np.array([
        3.1415926535897932, 2.7182818284590452, 1.1557273497909217179,
        0.86525597943226508722, 5.8598744820488384738, 0.42331082513074800310,
        8.5397342226735670655, 23.140692632779269006, 22.459157718361045473, 
        9.8696044010893586188,  7.3890560989306502272,  6.2831853071795864769, 
        5.4365636569180904707,  36.462159607207911771, 15.154262241479264190,
        114516.426867869998
    ])
# 变量占位符，形式为 #k
const_holder = np.array([f"#{i+1}" for i in range(len(values))])
const_num = len(const_holder)


# 规则表
# 为了优化计算速度生成的规则列表，支持嵌套0到8层

# 你也可以使用下面的Mathematica代码生成嵌套规则
# Rep[s_] := StringReplace[s, "E" -> "EES"];
# nl = NestList[Rep, "E", 8]
# "\""~StringJoin~#~StringJoin~"\"" & /@ nl

# E = 常数， S = 运算符

nest_rule_symbol = [
    "E",
    "EES",
    "EESEESS",
    "EESEESSEESEESSS",
    "EESEESSEESEESSSEESEESSEESEESSSS",
    "EESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSS",
    "EESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSSEESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSSS",
    "EESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSSEESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSSSEESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSSEESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSSSS",
    "EESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSSEESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSSSEESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSSEESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSSSSEESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSSEESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSSSEESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSSEESEESSEESEESSSEESEESSEESEESSSSEESEESSEESEESSSEESEESSEESEESSSSSSSS"
]

# 转换nest_rule为ndarray，E与S分别匹配1、0
nest_rule = np.array([ch for ch in nest_rule_symbol[opt.depth].replace("E", "1").replace("S", "0")], dtype=np.int32)
nest_rule_length = len(nest_rule)
print(nest_rule)


# cost映射
cost_dict = {
    "Pi": 1,
    "E": 1,
    "Plus": 3,
    "Times": 12,
    "Power": 72
}