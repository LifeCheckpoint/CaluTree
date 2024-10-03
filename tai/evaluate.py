import taichi as ti
from time import time

ti.init(arch=ti.cpu)

# Taichi 字段来存储表达式和结果
max_expr_length = 100  # 假设最大表达式长度为50
num_expressions = 10**6 

expr_field = ti.field(ti.i32, shape=(num_expressions, max_expr_length))
result_field = ti.field(ti.f64, shape=num_expressions)

# 创建一个全局的栈字段
stack_field = ti.field(ti.f64, shape=(num_expressions, max_expr_length))

@ti.func
def get_constant(token: ti.i32) -> ti.f64:
    return ti.select(token == 1, 3.2,
           ti.select(token == 2, 1.5,
           ti.select(token == 3, 4.7, 0.0)))  # 简化的常量映射

@ti.func
def evaluate_single_expr(expr: ti.template(), length: ti.i32, stack: ti.template(), i: ti.i32) -> ti.f64:
    stack_top = 0
    
    for j_r in range(0, length):
        j = length - j_r - 1 # 首项为 length - 1， 末项为 0
        token = expr[i, j]
        if token > 0:  # 常量
            stack[i, stack_top] = get_constant(token)
            stack_top += 1
        else:  # 操作符
            if stack_top >= 2:
                b, a = stack[i, stack_top - 1], stack[i, stack_top - 2]
                stack_top -= 2
                if token == -1:  # s1: 加法
                    stack[i, stack_top] = a + b
                elif token == -2:  # s2: 乘法
                    stack[i, stack_top] = a * b
                elif token == -3:  # s3: 除法
                    stack[i, stack_top] = a / b if b != 0 else 2e10  # 使用大数代替 inf
                stack_top += 1

    return stack[i, 0] if stack_top > 0 else 0.0

@ti.kernel
def evaluate_all_expressions():
    for i in range(num_expressions):
        length = 0
        while length < max_expr_length and expr_field[i, length] != 0:
            length += 1
        result_field[i] = evaluate_single_expr(expr_field, length, stack_field, i)

# 示例使用
def prepare_expressions():
    # 这里我们只准备一个表达式作为示例
    example_expr = [1, 2, -2, 2, 3, -1, -3]  # 对应 [#1, #2, s2, #2, #3, s1, s3]
    for j, token in enumerate(example_expr):
        expr_field[0, j] = token
    # 将相同的表达式复制到所有行
    for i in range(1, num_expressions):
        for j in range(len(example_expr)):
            expr_field[i, j] = expr_field[0, j]

# 运行评估
print("prepare...")
prepare_expressions()
t = time()
evaluate_all_expressions()
ti.sync()

# 获取结果
results = result_field.to_numpy()
print(time()-t)
print(f"First result: {results[0]}")