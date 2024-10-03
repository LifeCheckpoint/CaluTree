import sys
sys.path.append(".")

from tai.const import *
from tai.opt import opt
from time import time
import numpy as np
import taichi as ti

if opt.general.device.lower() == "cuda":
    ti.init(arch=ti.cuda)
else:
    ti.init(arch=ti.cpu)

# 读取opt到全局变量
num_expressions = opt.treeGenerate.num_expressions
batch = opt.treeGenerate.batch
prec_limit = opt.searching.precision_limit
target_number = opt.searching.target_number

# holder作为常数+符号占位符
holder = np.concatenate([const_holder, operators_symbol])

# 真实值映射
value_field = ti.field(ti.f64, shape=len(const_holder))
value_field.from_numpy(value_np)

# holder_index_field作为索引标号
holder_index_field = ti.field(ti.i32, shape=len(holder))
holder_index_field.from_numpy(np.array([i for i in range(len(holder))]))

# 转换 nest_rule(ndarray) 为 rule_field(field)
rule_field = ti.field(ti.i32, shape=nest_rule_length)
rule_field.from_numpy(nest_rule)

# 其它field定义
token_result_field = ti.field(ti.i32, shape=(num_expressions, nest_rule_length)) # token输出字段
stack_field = ti.field(ti.f64, shape=(num_expressions, nest_rule_length)) # 全局栈字段
calc_result_field = ti.field(ti.f64, shape=num_expressions) # 求值结果字段
eps_result_field = ti.field(ti.f64, shape=num_expressions) # 目标误差字段


# 计算表达式对应值
@ti.func
def evaluate_value_expr(i: int):
    stack_top = 0
    
    for j_r in range(0, nest_rule_length):
        j = nest_rule_length - j_r - 1 # 首项为 nest_rule_length - 1， 末项为 0
        token = token_result_field[i, j]
        if token < const_num:  # 常量
            stack_field[i, stack_top] = value_field[token] # 常量映射
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
                
                if stack_field[i, stack_top] > prec_limit or stack_field[i, stack_top] < -prec_limit:
                    stack_field[i, stack_top] = ti.math.inf
                    
                stack_top += 1

    calc_result_field[i] = stack_field[i, 0] if stack_top > 0 else ti.math.inf

# 计算表达式和目标值的差距
@ti.func
def evaluate_eps_expr(i: int):
    eps_result_field[i] = abs(calc_result_field[i] - target_number)

# 生成随机表达式并计算值
@ti.kernel
def generate_evaluate_random_expressions():
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
        evaluate_value_expr(i)

        # 求误差
        evaluate_eps_expr(i)


# 将索引转换为实际的字符串表达式
def exprindex_2_prefix(result_np):
    return [[holder[idx] for idx in expr] for expr in result_np]


# 测试用例
if __name__ == "__main__":
    # 运行生成器
    t = time()
    for bt in range(1000):
        generate_evaluate_random_expressions()
        ti.sync()

    # 将结果转换回NumPy数组并保存
    token_np = token_result_field.to_numpy()
    calc_np = calc_result_field.to_numpy()
    eps_np = eps_result_field.to_numpy()

    print(f"Time used {time() - t} s")
    print(f"expr_N = {len(calc_np)} × {batch}")

    # 转换为前缀表达式
    expressions = exprindex_2_prefix(token_np[:10])

    # 保存结果（这里只保存前10个作为示例）
    for i, expr in enumerate(expressions):
        print(f"Expression {i}: {' '.join(expr)}    |    Result: {calc_np[i]}    |    Eps:{eps_np[i]}")
