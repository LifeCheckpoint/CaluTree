# 该优化算法中不再使用多进程管理，仅有一个实例运行

class opt:

    class general:
        device = "cuda"

    cost_loop_N = 500 # 代价优化比较的大循环次数
    depth = 2 # 树深度
    enable_cost_algo = True # 启用cost优化方法（需要enable_optim_algo）
    enable_optim_algo = True # 启用数学优化方法（需要enable_wolfram）
    enable_wolfram = True # 使用Wolfram完成搜索算法
    eps = 0.1 # 搜索过滤精度
    final_eps = 0.00000001 # （用于cost优化）搜索过滤精度
    remove_low_quality_eps = True # （用于cost优化）是否丢弃低于final_eps的结果
    instant_output = True # 纯随机算法即时输出
    positive_eps = True # 要求搜索结果大于等于目标值
    num_iterations = 5000 # 单进程尝试建树数
    num_trying = 100 # 进程总数
    optim_eps = [0.1, 0.001, 0.001, 0.001] # 优化精度
    optim_loop_N = 4 # 优化大循环次数
    optim_output = False # 优化模式是否输出多个寻树结果
    precision_limit = 1e18 # 计算限度
    optim_limit = [1, 1e4] # 微分斜率大小限度
    target_number = 114514.1919810 # 目标数字
    the_first_N = 5 # 保留前N个结果
    wolfram_path = r"D:\Programs\Wolfram Mathematica 14.1\MathKernel.exe" # Wolfram MathKernel路径