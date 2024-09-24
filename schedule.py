from findtree import *

# 入口
if __name__ == "__main__":
    if opt.profiler:
        task_profiler_solo()
    else:
        if not opt.enable_optim_algo:
            task_calc_tree_without_optim()
        else:
            task_optim_calc_tree()

    if opt.enable_wolfram:
        wolf.stop()