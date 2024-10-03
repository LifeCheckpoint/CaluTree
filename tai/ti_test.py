import sys
sys.path.append(".")

import taichi as ti
import numpy as np
from time import time
from tai.const import *

ti.init(arch=ti.cuda)

# 创建holder作为常数+符号占位符，同时以holder_index_field作为索引标号
holder = np.concatenate([const_holder, operators_symbol])
holder_index_field = ti.field(ti.i32, shape=len(holder))
holder_index_field.from_numpy(np.array([i for i in range(len(holder))]))

# 转换 nest_rule(ndarray) 为 rule_field(field)
rule_field = ti.field(ti.i32, shape=nest_rule_length)
rule_field.from_numpy(nest_rule)

# 定义输出字段
num_expressions = 10**8 # 生成的表达式数量
result_field = ti.field(ti.i32, shape=(num_expressions, nest_rule_length))

# 生成随机表达式
@ti.kernel
def generate_random_expressions():
    for i in range(num_expressions):
        for j in range(nest_rule_length):
            # 常数
            if rule_field[j] == 1:
                index = ti.random(ti.i32) % const_num
                result_field[i, j] = holder_index_field[index]
            # 符号
            else:
                index = ti.random(ti.i32) % operators_symbol_num
                result_field[i, j] = holder_index_field[const_num + index] # 越过常数索引开始计数

# 将索引转换为实际的字符串表达式
def convert_to_expressions(result_np):
    return [[holder[idx] for idx in expr] for expr in result_np]

# 运行生成器
t = time()
for _ in range(10):
    generate_random_expressions()

# 将结果转换回NumPy数组并保存
result_np = result_field.to_numpy()
print(time()-t)



expressions = convert_to_expressions(result_np[:10])
ti.sync()

# 保存结果（这里只保存前10个作为示例）
for i, expr in enumerate(expressions):
    print(f"Expression {i}: {' '.join(expr)}")
