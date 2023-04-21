def plot_reward():
    import matplotlib
    import matplotlib.pyplot as plt
    import os
    file_list = os.listdir('figure/reward')
    valid_reward_all = []
    for i in file_list:
        f = open('figure/reward/'+i,'r')
        for line in f.readlines():
            if 'train_reward' in line:
                valid_reward_all.append(list(map(float,line.split(':')[1].split(','))))
        f.close()

    for count in range(len(file_list)):
        plt.plot(range(len(valid_reward_all[count])),
                       valid_reward_all[count])
        plt.title(file_list[count])
        plt.show()

def plot_time():
    import matplotlib
    import matplotlib.pyplot as plt

    # 读取文本文件中的数据
    f = open('figure/time.txt', 'r')
    data = f.read().split(',')
    # 绘制数据的图像
    plt.bar(range(len(data[:-1])), list(map(float, data[:-1])))
    # 设置图像的标题和坐标轴标签
    # plt.title('Data Plot')
    plt.xlabel('Subproblem index')
    plt.ylabel('Training time (s)')
    # 显示图像
    plt.show()

def plot_pf():
    import matplotlib.pyplot as plt
    import numpy as np
    f = open('figure/optimalvalue.txt', 'r')
    x = np.array([])
    y = np.array([])
    for line in f.readlines():
        data = line.replace('\n', '').split(',')
        data = list(map(float, data))
        x = np.append(x, data[0])
        y = np.append(y, data[1])
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
