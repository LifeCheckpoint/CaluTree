import sys
import os
from functools import wraps

def suppress_print(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 保存当前的标准输出
        original_stdout = sys.stdout
        
        try:
            # 重定向标准输出到 /dev/null (在Unix系统上)
            # 或 NUL (在Windows系统上)
            sys.stdout = open(os.devnull, 'w')
            
            # 调用原始函数
            result = func(*args, **kwargs)
        
        finally:
            # 恢复原始的标准输出
            sys.stdout.close()
            sys.stdout = original_stdout
        
        return result
    
    return wrapper