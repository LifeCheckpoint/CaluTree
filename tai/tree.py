import sys
sys.path.append(".")

from tai.const import *
from tai.const import const_holder
from tai.opt import opt
from time import time
from tqdm import tqdm
import gc
import numpy as np
import taichi as ti

if opt.general.device.lower() == "cuda":
    ti.init(arch=ti.cuda)
else:
    ti.init(arch=ti.cpu)


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
index_key_field = ti.field(dtype=ti.int32, shape=num_expressions) # 索引字段
index_key_field.from_numpy(np.arange(num_expressions))

# 最小值搜索器
@ti.data_oriented
class MinEpsIndexFinder:
    def __init__(self):
        self.min_eps = ti.field(dtype=ti.f64, shape=())
        self.min_index = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def find_min_eps_index(self, eps_field: ti.template(), index_field: ti.template()):
        # 初始化最小值为第一个元素
        self.min_eps[None] = eps_field[0]
        
        # 第一阶段：找到最小eps值
        for i in eps_field:
            ti.atomic_min(self.min_eps[None], eps_field[i])
        
        # 第二阶段：找到最小eps值对应的索引
        for i in eps_field:
            if eps_field[i] == self.min_eps[None]:
                self.min_index[None] = index_field[i]

# 计算表达式对应值
@ti.func
def evaluate_value_expr(i: int, prec_limit: float):
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
def evaluate_eps_expr(i: int, target_number: float):
    eps_result_field[i] = abs(calc_result_field[i] - target_number)

# 生成随机表达式并计算值
@ti.kernel
def generate_evaluate_random_expr_kernel(num_expressions: int, prec_limit: float, target_number: float):
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
        evaluate_value_expr(i, prec_limit)

        # 求误差
        evaluate_eps_expr(i, target_number)

# 寻找最小eps的表达式索引
def min_eps_index() -> int:    
    finder = MinEpsIndexFinder()
    finder.find_min_eps_index(eps_result_field, index_key_field)

    min_index = finder.min_index[None]
    
    return min_index

# 将索引转换为实际的字符串表达式
def exprindex_2_prefix(result_np):
    return [[holder[idx] for idx in expr] for expr in result_np]

# 整合 / 调度kernel函数
def generate_evaluate_random_expr(
        batch=opt.treeGenerate.batch, 
        num_expr=opt.treeGenerate.num_expressions, 
        prec_limit=opt.searching.precision_limit,
        target_number=opt.searching.target_number
    ):

    bar = tqdm(desc="生成计算", total=batch*num_expr)

    token_result = []
    calc_result = []
    eps_result = []

    for epoch in ti.static(range(batch)):
        # 进行计算
        generate_evaluate_random_expr_kernel(num_expressions=num_expr, prec_limit=prec_limit, target_number=target_number)
        ti.sync()

        # 将结果转换回NumPy数组并保存
        token_np = token_result_field.to_numpy()
        calc_np = calc_result_field.to_numpy()
        eps_np = eps_result_field.to_numpy()

        # 效率考虑，仅保留精度最高第一个结果
        index_best = min_eps_index()

        token_result.append(token_np[index_best])
        calc_result.append(calc_np[index_best])
        eps_result.append(eps_np[index_best])

        # print(eps_np[index_best])

        if (epoch + 1) % 20 == 0:
            gc.collect()

        bar.update(num_expr)
    
    bar.close()
    return token_result, calc_result, eps_result


# 测试用例，为了正常初始化需解除注释
    
tk, ca, ep = generate_evaluate_random_expr(
    batch=opt.treeGenerate.batch, 
    num_expr=num_expressions, 
    prec_limit=opt.searching.precision_limit, 
    target_number=opt.searching.target_number
)

# 转换为前缀表达式
expressions = exprindex_2_prefix(tk[:10])

# 保存结果（这里只保存前10个作为示例）
for i, expr in enumerate(expressions):
    print(f"Expression {i}: {' '.join(expr)}    |    Result: {ca[i]}    |    Eps:{ep[i]}")
