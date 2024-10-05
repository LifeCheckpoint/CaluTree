import sys
sys.path.append(".")

from tai.const import *
from tai.const import const_holder
from tai.opt import opt
from time import time
from utils import suppress_print
from tqdm import tqdm
import numpy as np
import taichi as ti

# 重置并初始化Taichi
@suppress_print
def reset_init_taichi():
    if opt.general.device.lower() == "cuda":
        arch = ti.cuda
    elif opt.general.device.lower == "vulkan":
        arch = ti.vulkan
    else:
        arch = ti.cpu

    ti.init(arch=arch, random_seed=int(time()))

# 最小值搜索器
@ti.data_oriented
class MinEpsIndexFinder:

    def __init__(self, expr_n):
        self.fb = ti.FieldsBuilder()

        self.min_eps = ti.field(dtype=ti.f64)
        self.min_index = ti.field(dtype=ti.i32)
        self.block_min_eps = ti.field(dtype=ti.f64)
        self.block_min_index = ti.field(dtype=ti.i32)

        self.fb.dense(ti.i, 1).place(self.min_eps)
        self.fb.dense(ti.i, 1).place(self.min_index)
        self.fb.dense(ti.i, expr_n).place(self.block_min_eps)
        self.fb.dense(ti.i, expr_n).place(self.block_min_index)

        self.st = self.fb.finalize()

    def __del__(self):
        self.st.destroy()

    @ti.kernel
    def find_min_eps_index(self, eps_field: ti.template(), index_field: ti.template()):
        n = eps_field.shape[0]
        
        # 每个线程处理的元素数量
        block_size = 256  # 可以根据实际情况调整
        num_blocks = (n + block_size - 1) // block_size
        
        # 第一阶段：并行计算每个块的最小值
        for block_id in range(num_blocks):
            self.min_eps[0] = 10000.0
            min_index = -1
            for i in range(block_id * block_size, min(n, (block_id + 1) * block_size)):
                if eps_field[i] < self.min_eps[0]:
                    self.min_eps[0] = eps_field[i]
                    min_index = index_field[i]
            self.block_min_eps[block_id] = self.min_eps[0]
            self.block_min_index[block_id] = min_index
        
        # 第二阶段：查找全局最小值
        ti.loop_config(serialize=True)
        for i in range(num_blocks):
            if i == 0 or self.block_min_eps[i] < self.min_eps[0]:
                self.min_eps[0] = self.block_min_eps[i]
                self.min_index[0] = self.block_min_index[i]

    # 获取结果
    def get_min_index(self) -> int:
        return self.min_index[0]

# 防止内存泄露，包裹至类中
@ti.data_oriented
class EvalTree:

    def __init__(self):
        # 统一管理fields内存与生命周期
        self.fb = ti.FieldsBuilder()

        # holder作为常数+符号占位符
        self.holder = np.concatenate([const_holder, operators_symbol])

        # 常数映射 field
        self.value_field = ti.field(ti.f64)
        self.fb.dense(ti.i, len(const_holder)).place(self.value_field)

        # 索引标号 holder_index_field
        self.holder_index_field = ti.field(ti.i32)
        self.fb.dense(ti.i, len(self.holder)).place(self.holder_index_field)

        # 树模板 rule_field
        self.rule_field = ti.field(ti.i32)
        self.fb.dense(ti.i, nest_rule_length).place(self.rule_field)
        
        # 生成token字段
        self.token_result_field = ti.field(ti.i32)
        self.fb.dense(ti.ij, (num_expressions, nest_rule_length)).place(self.token_result_field)
        
        # 栈字段
        self.stack_field = ti.field(ti.f64)
        self.fb.dense(ti.ij, (num_expressions, nest_rule_length)).place(self.stack_field)

        # 求值结果字段
        self.calc_result_field = ti.field(ti.f64)
        self.fb.dense(ti.i, num_expressions).place(self.calc_result_field)

        # 目标误差字段
        self.eps_result_field = ti.field(ti.f64)
        self.fb.dense(ti.i, num_expressions).place(self.eps_result_field)

        # 索引字段
        self.index_key_field = ti.field(dtype=ti.i32)
        self.fb.dense(ti.i, num_expressions).place(self.index_key_field)

        # 获得 SNode Tree
        self.snode_tree = self.fb.finalize()

        # 初始化 fields 值
        self.value_field.from_numpy(value_np)
        self.holder_index_field.from_numpy(np.array([i for i in range(len(self.holder))]))
        self.rule_field.from_numpy(nest_rule)
        self.index_key_field.from_numpy(np.arange(num_expressions))

    # 析构，结束该类fields生命周期，释放内存
    def __del__(self):
        self.snode_tree.destroy()

    # 计算表达式对应值
    @ti.func
    def evaluate_value_expr(self, i: int, prec_limit: float):
        stack_top = 0
        
        for j_r in range(0, nest_rule_length):
            j = nest_rule_length - j_r - 1 # 首项为 nest_rule_length - 1， 末项为 0
            token = self.token_result_field[i, j]
            if token < const_num:  # 常量
                self.stack_field[i, stack_top] = self.value_field[token] # 常量映射
                stack_top += 1
            else:  # 操作符
                if stack_top >= 2:
                    b, a = self.stack_field[i, stack_top - 1], self.stack_field[i, stack_top - 2]
                    stack_top -= 2

                    # 注意硬编码
                    real_symbol_token = token - const_num
                    if real_symbol_token == 0:  # s1: +
                        self.stack_field[i, stack_top] = a + b
                    elif real_symbol_token == 1:  # s2: -
                        self.stack_field[i, stack_top] = a - b
                    elif real_symbol_token == 2:  # s3: *
                        self.stack_field[i, stack_top] = a * b
                    elif real_symbol_token == 3:  # s4: /
                        self.stack_field[i, stack_top] = a / b if b != 0 else ti.math.inf  # 使用大数代替 inf
                    elif real_symbol_token == 4:  # s5: ^
                        self.stack_field[i, stack_top] = a ** b
                    elif real_symbol_token == 5:  # s6: I
                        self.stack_field[i, stack_top] = a
                    
                    if self.stack_field[i, stack_top] > prec_limit or self.stack_field[i, stack_top] < -prec_limit:
                        self.stack_field[i, stack_top] = ti.math.inf
                        
                    stack_top += 1

        self.calc_result_field[i] = self.stack_field[i, 0] if stack_top > 0 else ti.math.inf

    # 计算表达式和目标值的差距
    @ti.func
    def evaluate_eps_expr(self, i: int, target_number: float):
        self.eps_result_field[i] = abs(self.calc_result_field[i] - target_number)

    # 生成随机表达式并计算值
    @ti.kernel
    def generate_evaluate_random_expr_kernel(self, num_expressions: int, prec_limit: float, target_number: float):
        for i in range(num_expressions):

            # 生成表达式
            for j in range(nest_rule_length):
                # 常数
                if self.rule_field[j] == 1:
                    index = ti.random(ti.i32) % const_num
                    self.token_result_field[i, j] = self.holder_index_field[index]
                # 符号
                else:
                    index = ti.random(ti.i32) % operators_symbol_num
                    self.token_result_field[i, j] = self.holder_index_field[const_num + index] # 越过常数索引开始计数

            # 求值
            self.evaluate_value_expr(i, prec_limit)

            # 求误差
            self.evaluate_eps_expr(i, target_number)

    # 寻找最小eps的表达式索引
    def min_eps_index(self, expr_n: int) -> int:    
        finder = MinEpsIndexFinder(expr_n)
        finder.find_min_eps_index(self.eps_result_field, self.index_key_field)
        min_index = finder.get_min_index()
        del finder
        return min_index

    # 将索引转换为实际的字符串表达式
    def exprindex_2_prefix(self, result_np):
        return [[self.holder[int(idx)] for idx in expr] for expr in result_np]
    
    # 返回精度最高的计算结果
    def getFinalResult(self):
        # 效率考虑，仅保留精度最高第一个结果
        index_best = self.min_eps_index(num_expressions)

        # 取得最佳结果
        # 使用numpy切片会有严重的内存泄露，所以使用field循环赋值

        tidx, cidx, eidx = np.zeros((nest_rule_length, ), dtype=np.float64), 0.0, 0.0

        for i in range(nest_rule_length):
            tidx[i] = self.token_result_field[index_best, i]

        cidx = self.calc_result_field[index_best]
        eidx = self.eps_result_field[index_best]

        return tidx, cidx, eidx

# 整合 / 调度kernel函数
def generate_evaluate_random_expr(
        batch=opt.treeGenerate.batch, 
        num_expr=opt.treeGenerate.num_expressions, 
        prec_limit=opt.searching.precision_limit,
        target_number=opt.searching.target_number
    ):

    bar = tqdm(desc="Evaluating Started", total=batch*num_expr)

    token_result = []
    calc_result = []
    eps_result = []

    for epoch in range(batch):

        # 创建计算对象
        eval_tree = EvalTree()
        eval_tree.generate_evaluate_random_expr_kernel(num_expressions=num_expr, prec_limit=prec_limit, target_number=target_number)
        ti.sync()

        tidx, cidx, eidx = eval_tree.getFinalResult()
        token_result.append(tidx)
        calc_result.append(cidx)
        eps_result.append(eidx)
        ti.sync()

        # 清理计算对象
        del eval_tree

        bar.update(num_expr)
        bar.desc = f"Epoch {epoch+1}"

        # 定时重置ti防止内存问题累积
        if (epoch + 1) % 20 == 0:
            print("\nResetting Taichi...")
            reset_init_taichi()
    
    bar.close()
    return token_result, calc_result, eps_result


# 测试用例

reset_init_taichi()

tk, ca, ep = generate_evaluate_random_expr(
    batch=opt.treeGenerate.batch, 
    num_expr=num_expressions, 
    prec_limit=opt.searching.precision_limit, 
    target_number=opt.searching.target_number
)

# 转换为前缀表达式
tool_tree = EvalTree()
expressions = tool_tree.exprindex_2_prefix(tk[:10])
del tool_tree
reset_init_taichi()

# 保存结果（这里只保存前10个作为示例）
for i, expr in enumerate(expressions):
    print(f"Expression {i}: {' '.join(expr)}    |    Result: {ca[i]}    |    Eps:{ep[i]}")
