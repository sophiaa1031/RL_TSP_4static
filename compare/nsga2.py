import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.visualization.scatter import Scatter

# 定义优化问题
class MyProblem(get_problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=np.array([0, 0]), xu=np.array([1, 10]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0]**2
        f2 = (x[1]-2)**2
        out["F"] = [f1, f2]

# 定义算法并执行优化
problem = MyProblem()
algorithm = NSGA2(pop_size=100, n_offsprings=50, sampling=get_sampling("real_random"), crossover=get_crossover("real_sbx", prob=1.0, eta=3.0), mutation=get_mutation("real_pm", eta=3.0), eliminate_duplicates=True)
res = minimize(problem, algorithm, ('n_gen', 100), verbose=True)

# 打印结果
print(f"X1: {res.X[:, 0]}")
print(f"X2: {res.X[:, 1]}")
print(f"F1: {res.F[:, 0]}")
print(f"F2: {res.F[:, 1]}")

# 可视化结果
plot = Scatter()
plot.add(res.F, color="red")
plot.show()
