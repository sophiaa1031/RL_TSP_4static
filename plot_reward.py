import matplotlib
import matplotlib.pyplot as plt
import os
file_list = os.listdir('figure')
valid_reward_all = []
for i in file_list:
    f = open('figure/'+i,'r')
    for line in f.readlines():
        if 'valid_reward' in line:
            valid_reward_all.append(list(map(float,line.split(':')[1].split(','))))
    f.close()

for count in range(len(file_list)):
    plt.plot(range(len(valid_reward_all[count])),
                   valid_reward_all[count])
    plt.title(file_list[count])
    plt.show()
