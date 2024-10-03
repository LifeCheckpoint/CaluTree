# 该优化算法中线程由taichi统一调度，仅有一个实例运行
class opt:

    class general:
        enable_wolfram = True # 使用Wolfram
        enable_optim_algo = True # 使用优化算法（需要enable_wolfram）
        enable_cost_algo = True # 使用cost优化结果（需要enable_optim_algo）
        device = "cuda"
        wolfram_path = r"D:\Programs\Wolfram Mathematica 14.1\MathKernel.exe" # Wolfram MathKernel路径

    class treeGenerate:
        depth = 3
        num_expressions = 10**7 # 一次表达式生成数量
        batch = 10**4 # 一轮表达式生成次数
        keep_N = 5 # 每轮生成保留前N个结果

    class searching:
        target_number = 114514.1919810 # 搜索目标
        precision_limit = 1e20 # 计算精度

    class optimLoop:
        optim_loop_N = 4 # 优化循环次数（堆叠运算式层数）
        outside_optim_loop_N = 500 # 优化循环轮数
        optim_eqs = [0.1, 0.001, 0.001, 0.001] # 每次优化循环的搜索精度，如果是随机搜索，默认取首个
        final_eps = 0.00000001 # 最终精度要求（需要enable_cost_algo）
        remove_low_quality_eps = True # 是否丢弃低于final_eps的结果（需要enable_cost_algo）
        positive_eps = True # 要求搜索结果大于等于目标值
        optim_limit = [1, 1e4] # 微分斜率大小限度
