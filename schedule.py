from findtree import *
from wl import *
import os

# wolfram仅在主进程加载
if opt.enable_wolfram and mp.current_process().name == "MainProcess":
    wolf = wolfram()
else:
    wolf = None

# 入口
if __name__ == "__main__":
    if opt.profiler:
        task_profiler_solo()
    else:
        if not opt.enable_optim_algo:
            task_calc_tree_without_optim()
        else:
            if not opt.enable_cost_algo:
                task_optim_calc_tree()
            else:
                task_cost_optim_calc_tree()

    os._exit(0)