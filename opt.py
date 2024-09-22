class opt:
    depth = 6 # 树深度
    eps = 3 # 搜索过滤精度
    max_processes = 14 # 最大进程数
    num_iterations = 4 # 单进程尝试建树数
    num_trying = 10 # 进程总数
    profiler = False # 启用性能分析
    purpose_number = 114514 # 目标数字

    subtree_depth = 3 # 子树深度
    subtree_num = 300 # 尝试建立子树次数
    subtree_filter = lambda x: abs(x) < 1e7 # 子树过滤器，只有为True不会被筛除 