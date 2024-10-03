import taichi as ti
import numpy as np
from time import time

ti.init(arch=ti.cuda)

# 定义常数和运算符
constants = ["#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8"]
operators = ["s1", "s2", "s3", "s4", "s5", "s6"]

# 创建holder数组
holder = np.array(constants + operators + [""] * 6)  # 20个元素
holder_field = ti.field(ti.i32, shape=20)
holder_field.from_numpy(np.array([i for i in range(len(holder))]))

# 定义输出字段
num_expressions = 10**8 # 生成的表达式数量
expression_length = 20  # 每个表达式的长度
result_field = ti.field(ti.i32, shape=(num_expressions, expression_length))

@ti.kernel
def generate_random_expressions():
    for i in range(num_expressions):
        for j in range(expression_length):
            rand_int = ti.random(ti.i32)
            index = rand_int % 20
            result_field[i, j] = holder_field[index]

# 运行生成器
t = time()
for _ in range(10):
    generate_random_expressions()

# 将结果转换回NumPy数组并保存
result_np = result_field.to_numpy()
print(time()-t)

# 将索引转换为实际的字符串表达式
def convert_to_expressions(result_np):
    return [[holder[idx] for idx in expr] for expr in result_np]

expressions = convert_to_expressions(result_np[:10])
ti.sync()

# 保存结果（这里只保存前10个作为示例）
for i, expr in enumerate(expressions):
    print(f"Expression {i}: {' '.join(expr)}")
