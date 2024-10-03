import taichi as ti
import time

ti.init(arch=ti.cuda, random_seed=int(time.time()))

@ti.data_oriented
class PrefixExpressionGenerator:
    def __init__(self, depth):
        self.depth = depth
        self.stack = ti.field(dtype=ti.i32, shape=(2**(depth+1)))  # 栈大小与深度相关
        self.index = ti.field(dtype=ti.i32, shape=())
        self.operators = ti.static(ti.field(dtype=ti.i32, shape=(5)))  # 运算符索引
        self.operators[0] = 0  # '+'
        self.operators[1] = 1  # '-'
        self.operators[2] = 2  # '*'
        self.operators[3] = 3  # '/'
        self.operators[4] = 4  # '^'

    @ti.kernel 
    def generate(self):
        self.index[None] = 0
        self._generate_prefix_expression(self.depth)

    @ti.func
    def rand_choice_index(self, max_num): # [0, max_num - 1]
        return ti.random(ti.i32) % (max_num + 1)

    @ti.func
    def _generate_prefix_expression(self, depth):
        if depth == 0:
            self._push_leaf()
        else:
            # 随机选择运算符
            operator = self.rand_choice_index(4)  # 随机选择运算符的索引
            self._push_operator(operator)

            # 递归生成左右子树
            self._generate_prefix_expression(depth - 1)
            self._generate_prefix_expression(depth - 1)

    @ti.func
    def _push_operator(self, operator: int):
        self.stack[self.index[None]] = operator
        self.index[None] += 1

    @ti.func
    def _push_leaf(self):
        leaf_number = self.rand_choice_index(10)  # 随机叶子节点 #1, #2, ...
        self.stack[self.index[None]] = leaf_number + 10  # 确保叶子节点索引不与运算符冲突
        self.index[None] += 1

    def get_expression(self):
        return [self._to_string(self.stack[i]) for i in range(self.index[None])]

    @ti.func
    def _to_string(self, value: int) -> str:
        if value < 10:
            return f'#{value}'
        else:
            return self.operators[value - 10]  # 转换回运算符

# 使用示例
depth = 3  # 指定深度
generator = PrefixExpressionGenerator(depth)
generator.generate()
expression = generator.get_expression()
print(expression)
