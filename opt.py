class opt:
    depth = 3 # 树深度
    enable_optim_algo = True # 启用数学优化方法（需要enable_wolfram）
    enable_wolfram = True # 使用Wolfram完成搜索算法
    eps = 0.1 # 搜索过滤精度
    instant_output = True # 纯随机算法即时输出
    optim_output = False # 优化模式是否输出多个寻树结果
    max_processes = 10 # 最大进程数
    num_iterations = 5000 # 单进程尝试建树数
    num_trying = 100 # 进程总数
    optim_loop_N = 4 # 优化大循环次数
    optim_eps = [0.1, 0.01, 0.01, 0.01] # 优化精度
    precision_limit = 1e15 # 计算限度
    profiler = False # 启用性能分析
    target_number = 114514.1919810 # 目标数字
    the_first_N = 5 # 保留前N个结果
    wolfram_path = r"D:\Programs\Wolfram Mathematica 14.1\MathKernel.exe" # Wolfram MathKernel路径