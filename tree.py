import warnings
import traceback
import torch.profiler
import random
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from operator import add, sub, mul, pow
from numba import jit
from itertools import product
from anytree import Node, RenderTree

warnings.simplefilter("ignore", RuntimeWarning)

# 定义运算符字典, 有数值错误会抛出到顶层
operators = {
    "1": add,
    "2": sub,
    "3": mul,
    "4": lambda x, y: x / y,
    "5": lambda x, y: pow(x, y),
    "6": lambda x: x
}

# 自动生成随机计算树结构
def create_random_tree(operators_list, leaves_list, depth, parent=None):
    if depth == 0:
        # 生成随机叶子节点
        leaf = random.choice(leaves_list)
        return Node(leaf, parent=parent)

    # 随机选择运算符节点
    operator = random.choice(operators_list)
    node = Node(operator, parent=parent)

    # 递归生成左右子树
    create_random_tree(operators_list, leaves_list, depth-1, node)
    create_random_tree(operators_list, leaves_list, depth-1, node)

    return node

# 生成多个随机树
def generate_random_trees(operators, leaves, num_trees, max_depth):
    all_trees = []
    for _ in range(num_trees):
        tree = create_random_tree(list(operators.keys()), leaves, max_depth)
        all_trees.append(tree)
    return all_trees

# 使用 JIT 加速的递归计算函数
@jit(nopython=True)
def compute_tree_jit(node_name, left, right, values):
    if node_name.startswith("#"):
        return values[int(node_name[1:]) - 1]  # 取出#1, #2的实际值
    elif node_name.startswith("6"):
        return float(node_name.split("_")[1])  # 提取常量值
    return left, right

# 计算树结果
def safe_compute_tree(node, values):
    if node.name.startswith("#"):
        return values[int(node.name[1:]) - 1]  # 获取操作数的值
    elif node.name.startswith("6") and "_" in node.name:
        return float(node.name.split("_")[1])  # 提取常量值

    # 递归计算子节点
    left_value = safe_compute_tree(node.children[0], values) if len(node.children) > 0 else 0
    right_value = safe_compute_tree(node.children[1], values) if len(node.children) > 1 else 0

    if node.name == "6":  # 一元运算符
        result = operators[node.name](left_value)
        if abs(result) > 1e15:
            raise ValueError()
        return result
    else:  # 二元运算符
        result = operators[node.name](left_value, right_value)
        if abs(result) > 1e15:
            raise ValueError()
        return result

def tree_to_expression(node):
    if node.name.startswith("#"):
        return node.name  # 直接返回 #1 和 #2

    left_expr = tree_to_expression(node.children[0]) if len(node.children) > 0 else ""
    right_expr = tree_to_expression(node.children[1]) if len(node.children) > 1 else ""

    if node.name == "1":
        return f"({left_expr} + {right_expr})"
    elif node.name == "2":
        return f"({left_expr} - {right_expr})"
    elif node.name == "3":
        return f"({left_expr} * {right_expr})"
    elif node.name == "4":
        return f"({left_expr} / {right_expr})"
    elif node.name == "5":
        return f"({left_expr} ^ {right_expr})"
    elif node.name == "6":
        return f"{left_expr}"  # 处理常量

values = np.array([
    3.1415926535897932, # pi
    2.7182818284590452, # e
    1.1557273497909217, # pi/e
    0.8652559794322651, # e/pi
    5.8598744820488385, # pi+e
    0.4233108251307480, # pi-e
    8.5397342226735671, # pi*e
    23.140692632779269, # e^pi
    22.459157718361045 # pi^e
], dtype=np.float64)
placeholder = [f"#{i+1}" for i in range(len(values))]

def generate_tree_choices(num_trees, max_depth, propose, eps):
    
    trees = generate_random_trees(operators, placeholder, num_trees=num_trees, max_depth=max_depth)

    result_trees = []
    # 遍历生成的树并计算结果
    for _, tree in enumerate(trees):
        try:
            result = safe_compute_tree(tree, values)
        except:
            result = float('inf')

        if abs(result - propose) < np.float64(eps):
            result_trees.append((tree, result))

    return result_trees

depth = 6
eps = 3
max_processes = 14
num_iterations = 4
num_trying = 10
purpose_number = 114514
profiler = False

def worker_findtree(_):
    try:
        return generate_tree_choices(num_trying, depth, purpose_number, eps)
    except Exception as e:
        print(f"Error in worker: {e}")
        traceback.print_exc()
        return []

if __name__ == "__main__":
    if profiler:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(
                wait=1,      # 等待1个步骤后开始记录
                warmup=1,    # 热身1个步骤
                active=2     # 在接下来的2个步骤中记录
            ),
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs')
        ) as prof:
            for _ in range(4):
                worker_findtree(1)
                prof.step()
    else:
        # FIND
        with mp.Pool(processes=max_processes) as pool:
            pbar = tqdm(total=num_iterations*num_trying)
            trees = []

            for result in pool.imap(worker_findtree, iterable=range(num_iterations)):
                if len(result) > 0:
                    for tree, calc_result in result:
                        print(f"Expression {tree_to_expression(tree)}\nCalu {calc_result}")

                    trees.extend(result)
                pbar.update(num_trying)
                pbar.set_description(f"Current Result: {len(trees)}")

            pbar.close()

        for tree, calc_result in trees:
            print(f"Tree:")
            # for pre, _, node in RenderTree(tree):
            #     print(f"{pre}{node.name}")
            print("计算结果:", calc_result)
            print("表达式：", tree_to_expression(tree))
            print("\n")
        
        if len(trees) == 0:
            print("未找到目标")