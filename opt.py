class opt:
    depth = 2 # 树深度
    eps = 0.001 # 搜索过滤精度
    max_processes = 10 # 最大进程数
    num_iterations = 2000 # 单进程尝试建树数
    num_trying = 5000 # 进程总数
    profiler = False # 启用性能分析
    purpose_number = -5.31378501 # 目标数字

    subtree_depth = 3 # 子树深度
    subtree_num = 300 # 尝试建立子树次数
    subtree_filter = lambda x: abs(x) < 1e7 # 子树过滤器，只有为True不会被筛除 