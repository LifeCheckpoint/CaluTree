import warnings
import traceback
import random
import numpy as np
from typing import Tuple
from opt import *
from preprocess import *
from anytree import Node

warnings.simplefilter("ignore", RuntimeWarning)

# calcTree仍然用传统的递归计算方法，因为wl的速度不会比原生计算速度高

class calcTree:
    def __init__(self, operators=using_operators, placeholder=placeholder):
        self.operators = operators
        self.operators_list = list(self.operators.keys())
        self.placeholder = placeholder

    # 自动生成随机计算树结构
    def create_random_tree(self, leaves_list, depth, parent=None) -> Node:
        if depth == 0:
            # 生成随机叶子节点
            leaf = random.choice(leaves_list)
            return Node(leaf, parent=parent)

        # 随机选择运算符节点
        operator = random.choice(self.operators_list)
        node = Node(operator, parent=parent)

        # 递归生成左右子树
        self.create_random_tree(leaves_list, depth-1, node)
        self.create_random_tree(leaves_list, depth-1, node)

        return node

    # 生成多个随机树
    def generate_multi_random_trees(self, num_trees, max_depth):
        all_trees = []
        for _ in range(num_trees):
            tree = self.create_random_tree(self.placeholder, max_depth)
            all_trees.append(tree)
        return all_trees

    def extract_value_jit(self, node_name, values):
        if node_name.startswith("#"):
            return values[int(node_name[1:]) - 1]  # 取出#1, #2的实际值
        elif node_name.startswith("6") and "_" in node_name:
            return float(node_name.split("_")[1])  # 提取常量值
        else:
            return True

    # 计算树Node求值
    def safe_compute_tree(self, node, values):
        ex_value = self.extract_value_jit(node.name, values)

        # 若已到达常量位置
        if ex_value != True:
            return ex_value

        # 递归计算子节点
        left_value = self.safe_compute_tree(node.children[0], values) if len(node.children) > 0 else 0
        right_value = self.safe_compute_tree(node.children[1], values) if len(node.children) > 1 else 0

        if node.name == "6":  # 恒等（一元）
            result = self.operators[node.name](left_value)
        else:  # 二元运算符
            result = self.operators[node.name](left_value, right_value)
        
        if abs(result) > opt.precision_limit:
            raise ValueError()
        return result

    # 将计算树转化为形式公式
    def tree_to_expression(self, node):
        if node.name.startswith("#"):
            return node.name  # 直接返回常量

        left_expr = self.tree_to_expression(node.children[0]) if len(node.children) > 0 else ""
        right_expr = self.tree_to_expression(node.children[1]) if len(node.children) > 1 else ""

        # + - * / ^
        if int(node.name)-1 in range(5):
            op = ["+", "-", "*", "/", "^"][int(node.name)-1]
            return f"({left_expr} {op} {right_expr})"
        elif node.name == "6": # 恒等函数
            return f"{left_expr}"

def generate_tree_to_purpose(num_trees, max_depth, purpose, eps) -> Tuple[Node, float, float]:
    tr = calcTree()
    trees = tr.generate_multi_random_trees(num_trees=num_trees, max_depth=max_depth)
    result_trees = []

    # 遍历生成的树并安全求值
    for tree in trees:
        try:
            result = tr.safe_compute_tree(tree, values)
        except Exception as e:
            # print(e)
            continue
        
        delta = abs(result - purpose)
        if delta < eps: result_trees.append((tree, result, delta))

    return result_trees

def worker_findtree(_):
    try:
        print(opt.target_number)
        return generate_tree_to_purpose(opt.num_trying, opt.depth, opt.target_number, opt.eps)
    except Exception as e:
        print(f"Error occured in worker: {e}")
        traceback.print_exc()
        return []

def worker_findtree_dynamic(target_number, eps, value):
    values = value
    try:
        return generate_tree_to_purpose(opt.num_trying, opt.depth, target_number, eps)
    except Exception as e:
        print(f"Error occured in worker: {e}")
        traceback.print_exc()
        return []