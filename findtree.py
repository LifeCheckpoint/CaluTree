import torch.profiler
import multiprocessing as mp
import os
from typing import Tuple
from itertools import chain
from tqdm import tqdm
from tree import *
from opt import *
from wl import *

# wolfram仅在主进程加载
if opt.enable_wolfram and mp.current_process().name == "MainProcess":
    wolf = wolfram()
else:
    wolf = None



# 随机找树（用于优化）
def calc_tree_optim():
    print(f"Searching... Total {opt.num_iterations * opt.num_trying}")
    tr = calcTree()

    with mp.Pool(processes=opt.max_processes) as pool:
        trees = []
        param = [(opt.target_number, opt.eps, values) for _ in range(opt.num_iterations)]
        result = pool.starmap(worker_findtree_dynamic, param)
        trees = list(chain.from_iterable(result))

    # 按delta从小到大排序并保留一定数量
    trees.sort(key=lambda res: res[2])
    trees = trees[:opt.the_first_N]

    if opt.optim_output:
        print("------ 最终结果 ------")
        for i, (tree, calc_result, delta) in enumerate(reversed(trees)):
            print(f"-----Tree {i}\nExpression {tr.tree_to_expression(tree)}\nCalculated Result {calc_result}\nDelta {delta}")
    if len(trees) == 0:
        print("未找到目标")

    return trees

# 优化信息计算，返回对应算式与下一目标
def get_optim_info(tree, final) -> Tuple[str, float]:
    tr = calcTree()
    # 将树对应转换为对应算式，然后得到其变化率列表后找到其变化率最小的一个式子。
    origin_expr = tr.tree_to_expression(tree[0])
    wolf.wolfram_evaluate(f"optimalExpr=First@First@DeltaTest[ReplaceExpr[{origin_expr}]];")
    current_tree_expression_x = str(wolf.wolfram_evaluate("optimalExpr")).replace("Global`", "")

    info = str(wolf.wolfram_evaluate("optimalExpr"))
    print(f"优化 | Current formula = {info}")

    # 尝试反求该式求值目标
    try:
        new_target = float(wolf.wolfram_evaluate(f"InverseSolve[optimalExpr,{final}]"))
    except:
        raise ValueError("反求错误，更换其它树")
    
    if abs(new_target) < 1 or abs(new_target) > 1e4:
        raise ValueError("目标大小超限，更换其它树")
    
    return current_tree_expression_x, new_target

# 优化找树主任务循环
def task_optim_calc_tree():
    tr = calcTree()
    tree_expression_x = [] # 含有 x 变元的表达式
    current_target = opt.target_number

    for loop_i in range(opt.optim_loop_N):

        print(f"---------- epoch {loop_i+1} ----------")

        # 通过动态设置opt来调整寻找方案
        opt.target_number = current_target
        opt.eps = opt.optim_eps[loop_i]
        print(f"当前优化目标：{opt.target_number}\n当前目标精度要求：{opt.eps}")

        # 搜索
        while True:
            trees = calc_tree_optim()
            if len(trees) != 0:

                # 到达最后一轮，无需进行优化
                if loop_i == opt.optim_loop_N - 1:
                    tree_expression_x.append(tr.tree_to_expression(trees[0][0])) # 注意最后一个表达式不含x
                    break

                suc = False
                
                # 循环选取树种
                for i, best_tree in enumerate(trees):
                    try:
                        # 尝试优化该树
                        expr, tar = get_optim_info(best_tree, current_target)
                        current_target = tar
                        tree_expression_x.append(expr)
                        suc = True
                        break
                    except:
                        # 优化失败
                        continue
                        
                # 检查优化结果
                if suc:
                   break
                else:
                    print("当前搜索未寻到可供优化算式，将会重试") 

            # 没有搜索到树
            else:
                print("未能在当前精度找到目标算式，将会重试")
        
    # 完成优化大循环，迭代所有式子
    current_formula = "x"
    for nest_formula in tree_expression_x:
        current_formula = current_formula.replace("x", nest_formula)
    current_formula = wolf.wolfram_evaluate(f"ReplaceExpr[{current_formula}]")
    value_formula = str(wolf.wolfram_evaluate(f"ExprN[{current_formula},30]"))

    print(f"最终结果：\nExpression: {current_formula}\nValue: {value_formula}")
    os._exit(0)



# 单树性能测试
def task_profiler_solo():
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs')
    ) as prof:
        for _ in range(4):
            worker_findtree_dynamic(1)
            prof.step()



# 随机找树，无优化
def task_calc_tree_without_optim():
    tr = calcTree()

    with mp.Pool(processes=opt.max_processes) as pool:
        pbar = tqdm(total=opt.num_iterations*opt.num_trying, desc="Loading...")
        trees = []

        for result in pool.imap(worker_findtree_dynamic, iterable=range(opt.num_iterations)):

            # 即时输出
            if opt.instant_output:
                for tree, calc_result, delta in result:
                    print(f"-----\nExpression {tr.tree_to_expression(tree)}\nCalculated Result {calc_result}\nDelta {delta}")

            trees.extend(result)
            pbar.update(opt.num_trying)
            pbar.set_description(f"Current Result: {len(trees)}")

        pbar.close()

    # 按delta从小到大排序并保留一定数量
    trees.sort(key=lambda res: res[2])
    trees = trees[:opt.the_first_N]

    print("------ 最终结果 ------")
    for i, (tree, calc_result, delta) in enumerate(reversed(trees)):
        print(f"-----Tree {i}\nExpression {tr.tree_to_expression(tree)}\nCalculated Result {calc_result}\nDelta {delta}")
    if len(trees) == 0:
        print("未找到目标")

    return trees