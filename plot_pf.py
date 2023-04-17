import matplotlib.pyplot as plt
import numpy as np

# 定义五个点的坐标
x = np.array([0.524976909160614, 0.1264934241771698, 0.4035329818725586, 0.4618919789791107, 0.4961088001728058, 0.5105521082878113])  # 第一个目标函数的坐标
y = np.array([0.2663421332836151, 0.4726107120513916, 0.49224403500556946, 0.469144344329834, 0.3667467534542084, 0.2662496268749237])  # 第二个目标函数的坐标

# 使用argsort函数按照x从小到大的顺序排序索引值
sorted_index = np.argsort(x)

# 重新排列x和y
x_sorted = x[sorted_index]
y_sorted = y[sorted_index]

# 绘制散点图
plt.plot(x_sorted, y_sorted)

# 添加轴标签和标题
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Pareto Front')

# 显示图形
plt.show()
