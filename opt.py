class opt:
    depth = 2 # 树深度
    eps = 0.1 # 搜索过滤精度
    max_processes = 10 # 最大进程数
    num_iterations = 100 # 单进程尝试建树数
    num_trying = 50 # 进程总数
    profiler = False # 启用性能分析
    purpose_number = -5.31378501 # 目标数字
    precision_limit = 1e15 # 计算限度
    wolfram_using = True # 使用Wolfram完成搜索算法
    wolfram_path = r"D:\Programs\Wolfram Mathematica 14.1\MathKernel.exe" # Wolfram MathKernel路径