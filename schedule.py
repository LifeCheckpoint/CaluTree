import torch.profiler
import multiprocessing as mp
from tqdm import tqdm
from tree import *
from opt import *

def task_calc_tree_multi():
    tr = calcTree()

    with mp.Pool(processes=opt.max_processes) as pool:
        pbar = tqdm(total=opt.num_iterations*opt.num_trying)
        trees = []

        for result in pool.imap(worker_findtree, iterable=range(opt.num_iterations)):
            if len(result) > 0:
                for tree, calc_result in result:
                    print(f"Expression {tr.tree_to_expression(tree)}\nCalu {calc_result}")

                trees.extend(result)
            pbar.update(opt.num_trying)
            pbar.set_description(f"Current Result: {len(trees)}")

        pbar.close()

    for tree, calc_result in trees:
        print(f"Tree:")
        # for pre, _, node in RenderTree(tree):
        #     print(f"{pre}{node.name}")
        print("计算结果:", calc_result)
        print("表达式：", tr.tree_to_expression(tree))
        print("\n")
    
    if len(trees) == 0:
        print("未找到目标")

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

if __name__ == "__main__":
    if opt.profiler:
        task_profiler_solo()
    else:
        task_calc_tree_multi()