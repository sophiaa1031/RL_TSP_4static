"""Defines the main task for the TSP

The TSP is defined by the following traits:
    1. Each city in the list must be visited once and only once
    2. The salesman must return to the original node at the end of the tour

Since the TSP doesn't have dynamic elements, we return an empty list on
__getitem__, which gets processed in trainer.py to be None

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TSPDataset(Dataset):  # 初始化生成训练数据

    def __init__(self, episode=1e6, num_cars=10, iteration=20, seed=None):
        super(TSPDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(123456789)

        np.random.seed(seed)
        torch.manual_seed(seed)
        pwr = torch.ones((episode, num_cars, 1, iteration+1)) * 0.1  # 0.09*torch.rand((episode, num_cars, 1, iteration)) + 0.01  #p [0.01,0.1]
        fre = torch.ones((episode, num_cars, 1, iteration+1)) * 2  # 1*torch.rand((episode, num_cars, 1, iteration)) + 2  #f [2,3]
        vel = 5 * torch.rand((episode, num_cars, 1, iteration+1)) + 15  # v [15, 20]
        rho = torch.rand((episode, num_cars, 1, iteration+1))
        rho = F.softmax(rho, dim=1)
        self.static = torch.cat([pwr, fre, vel, rho], 2) #（samples, cars number, (q,f,v)  iteration）

        latency_itr = torch.zeros(episode, num_cars, 1, iteration+1)
        travel_dis = torch.zeros(episode, num_cars, 1, iteration+1)
        rsu_dis = torch.ones(episode, num_cars, 1, iteration+1)*301 # math.sqrt(300**2+10_num_cars**2)
        self.dynamic = torch.cat([latency_itr, travel_dis, rsu_dis], 2)  #（samples, (latency, travel distance, distance), cars number）
        self.num_cars = num_cars
        self.size = episode


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.static[idx], self.dynamic[idx], [])


def update_mask(mask, dynamic, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask


def reward(static, dynamic, action, obj1_scaling,obj2_scaling, w1=1, w2=0):
    """
    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. x, y) data
    action: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route. of size
    (batch_size, num_cities)
    """
    #static  [batch_size,num_cars,features,iteration]
    # dynamic  [batch_size,num_cars,features,iteration]
    # action  [batch_size,num_cars,action,iteration-1]
    batch_size, num_cars, _, round = static.shape
    iteration = round - 1
    obj1 = torch.zeros([batch_size,iteration]).to(device) #
    obj2 = torch.zeros([batch_size, iteration]).to(device) #
    obj = torch.zeros([batch_size, iteration]).to(device)
    for iter in range(iteration):
        dis = dynamic[:, :, 2, iter]
        rate = action[:, :, 2, iter]*10*torch.log2(1+1e7*static[:,:,0,iter]*torch.pow(dis, -2))
        if iter == 0:
            obj1[:, iter] = torch.sum(static[:, :, 3, iter]/
                                               torch.pow(torch.pow(2, action[:, :, 1, iter]) - 1,2), dim=1)
            obj2_temp = 0.002*action[:, :, 1, iter]/static[:, :, 1, iter] + \
               action[:, :, 1, iter]/32/rate
            obj2[:, iter],idx = torch.max(obj2_temp,dim=1)

        else:
            obj1[:, iter] = obj1[:, iter-1] + torch.sum(static[:, :, 3, iter]/
                                               torch.pow(torch.pow(2, action[:, :, 1, iter]) - 1,2), 1)
            obj2_temp = 0.002*action[:, :, 1, iter]/static[:, :, 1, iter] + \
              action[:, :, 1, iter]/32/rate
            obj2[:, iter], idx = torch.max(obj2_temp, dim=1)
            obj2[:, iter] = obj2[:, iter] + obj2[:, iter - 1]
    #     obj1[batch_size,iter] = obj1+torch.sum(action[:,:,0,:]/(2^action[:,:,1,:]-1)^2

        #obj2 = obj2+torch.max(action[0]*0.005*action[1]/static[1] + action[0]*0.01*action[1]/snr.squeeze(),dim=1)
    if torch.max(obj1) > obj1_scaling.max:
        obj1_scaling.set_max(torch.max(obj1))
    if torch.min(obj1) < obj1_scaling.min:
        obj1_scaling.set_min(torch.min(obj1))

    if torch.max(obj2) > obj2_scaling.max:
        obj2_scaling.set_max(torch.max(obj2))
    if torch.min(obj2) < obj2_scaling.min:
        obj2_scaling.set_min(torch.min(obj2))

    for iter in range(iteration):
        obj1[:, iter] = (obj1[:, iter] - obj1_scaling.min)/(obj1_scaling.max-obj1_scaling.min)
        obj2[:, iter] = (obj2[:, iter] - obj2_scaling.min) / (obj2_scaling.max - obj2_scaling.min)
        obj[:,iter] = -w1*obj1[:, iter] - w2*obj2[:, iter] + 1e-2/num_cars*(torch.sum(rate-1, dim=1))
    return obj.detach(), obj1, obj2



def render(static, action, save_path):
    """Plots the found tours."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(action))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = action[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        # End tour at the starting index
        idx = idx.expand(static.size(1), -1)
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)

        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        #plt.subplot(num_plots, num_plots, i + 1)
        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=400)


class RewardScaling:
    def __init__(self,max_=-999,min_=999):
        self.max = max_
        self.min = min_

    def set_max(self,value):
        self.max = value

    def set_min(self,value):
        self.min = value

