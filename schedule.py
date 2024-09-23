import torch.profiler
import multiprocessing as mp
from tqdm import tqdm
from tree import *
from opt import *
from wl import *

if opt.wolfram_using:
    wolf = wolfram()
else:
    wolf = None

def task_calc_tree_multi():
    tr = calcTree()

    with mp.Pool(processes=opt.max_processes) as pool:
        pbar = tqdm(total=opt.num_iterations*opt.num_trying, desc="Loading...")
        trees = []

        for result in pool.imap(worker_findtree, iterable=range(opt.num_iterations)):

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

# 用于性能测试
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
            worker_findtree(1)
            prof.step()

# 入口
if __name__ == "__main__":
    if opt.profiler:
        task_profiler_solo()
    else:
        task_calc_tree_multi()

    if opt.wolfram_using:
        wolf.stop()