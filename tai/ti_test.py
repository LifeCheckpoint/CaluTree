import sys
sys.path.append(".")

import taichi as ti
import numpy as np
from time import time
from tai.const import *

ti.init(arch=ti.cuda)

num_expressions = 10**7 # 生成的表达式数量
batch = 10**4

# 创建holder作为常数+符号占位符，同时以holder_index_field作为索引标号
holder = np.concatenate([const_holder, operators_symbol])

value_field = ti.field(ti.f64, shape=len(const_holder))
value_field.from_numpy(value_np)

holder_index_field = ti.field(ti.i32, shape=len(holder))
holder_index_field.from_numpy(np.array([i for i in range(len(holder))]))

# 转换 nest_rule(ndarray) 为 rule_field(field)
rule_field = ti.field(ti.i32, shape=nest_rule_length)
rule_field.from_numpy(nest_rule)

# token输出字段
token_result_field = ti.field(ti.i32, shape=(num_expressions, nest_rule_length))

# 全局栈字段
stack_field = ti.field(ti.f64, shape=(num_expressions, nest_rule_length))

# 求值结果字段
calc_result_field = ti.field(ti.f64, shape=num_expressions)

# 常量映射函数，给定token编号返回常量
@ti.func
def get_constant(token: ti.i32) -> ti.f64:
    return value_field[token]

# 计算表达式对应值
@ti.func
def evaluate_single_expr(i: int):
    stack_top = 0
    
    for j_r in range(0, nest_rule_length):
        j = nest_rule_length - j_r - 1 # 首项为 nest_rule_length - 1， 末项为 0
        token = token_result_field[i, j]
        if token < const_num:  # 常量
            stack_field[i, stack_top] = get_constant(token)
            stack_top += 1
        else:  # 操作符
            if stack_top >= 2:
                b, a = stack_field[i, stack_top - 1], stack_field[i, stack_top - 2]
                stack_top -= 2

                # 注意硬编码
                real_symbol_token = token - const_num
                if real_symbol_token == 0:  # s1: +
                    stack_field[i, stack_top] = a + b
                elif real_symbol_token == 1:  # s2: -
                    stack_field[i, stack_top] = a - b
                elif real_symbol_token == 2:  # s3: *
                    stack_field[i, stack_top] = a * b
                elif real_symbol_token == 3:  # s4: /
                    stack_field[i, stack_top] = a / b if b != 0 else ti.math.inf  # 使用大数代替 inf
                elif real_symbol_token == 4:  # s5: ^
                    stack_field[i, stack_top] = a ** b
                elif real_symbol_token == 5:  # s6: I
                    stack_field[i, stack_top] = a
                
                if stack_field[i, stack_top] > 2e10 or stack_field[i, stack_top] < -2e10:
                    stack_field[i, stack_top] = ti.math.inf
                    
                stack_top += 1

    calc_result_field[i] = stack_field[i, 0] if stack_top > 0 else ti.math.inf


# 生成随机表达式并计算值
@ti.kernel
def generate_calculate_random_expressions():
    for i in range(num_expressions):

        # 生成表达式
        for j in range(nest_rule_length):
            # 常数
            if rule_field[j] == 1:
                index = ti.random(ti.i32) % const_num
                token_result_field[i, j] = holder_index_field[index]
            # 符号
            else:
                index = ti.random(ti.i32) % operators_symbol_num
                token_result_field[i, j] = holder_index_field[const_num + index] # 越过常数索引开始计数

        # 求值
        evaluate_single_expr(i)


# 将索引转换为实际的字符串表达式
def convert_to_expressions(result_np):
    return [[holder[idx] for idx in expr] for expr in result_np]

# 运行生成器
t = time()
for bt in range(1000):
    generate_calculate_random_expressions()
    ti.sync()

# 将结果转换回NumPy数组并保存
token_np = token_result_field.to_numpy()
calc_np = calc_result_field.to_numpy()
print(f"Time used {time() - t} s")
print(f"expr_N = {len(calc_np)} × {batch}")

# 转换为前缀表达式
expressions = convert_to_expressions(token_np[:10])

# 保存结果（这里只保存前10个作为示例）
for i, expr in enumerate(expressions):
    print(f"Expression {i}: {' '.join(expr)}    |    Result: {calc_np[i]}")
