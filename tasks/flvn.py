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


class TSPDataset(Dataset):

    def __init__(self, num_cars=20, loop=1e6, iteration=10, seed=None):
        super(TSPDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(123456789)

        np.random.seed(seed)
        torch.manual_seed(seed)
        p = 0.09*torch.rand((loop, num_cars, 1, iteration)) + 0.01  #p [0.01,0.1]
        f = 1*torch.rand((loop, num_cars, 1, iteration)) + 2  #f [2,3]
        v = 20 * torch.rand((loop, num_cars, 1, iteration)) + 60  # v [60, 80]
        self.dataset = torch.cat([p, f, v], 2) #（samples, cars number, (q,f,v)  iteration）
        self.dynamic = torch.zeros(loop, num_cars, 3, iteration)  #（samples, (latency, travel distance, distance), cars number）
        self.num_cars = num_cars
        self.size = loop


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx], self.dynamic[idx], [])


def update_mask(mask, dynamic, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask


def reward(static, tour_indices, w1=1, w2=0):
    """
    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. x, y) data
    tour_indices: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route. of size
    (batch_size, num_cities)
    """

    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Make a full tour by returning to the start
    y = torch.cat((tour, tour[:, :1]), dim=1)
    # first 2 is xy coordinate, third column is another obj
    y_dis = y[:, :, :2]
    y_dis2 = y[:, :, 2:]

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y_dis[:, :-1] - y_dis[:, 1:], 2), dim=2))
    obj1 = tour_len.sum(1).detach()

    tour_len2 = torch.sqrt(torch.sum(torch.pow(y_dis2[:, :-1] - y_dis2[:, 1:], 2), dim=2))
    obj2 = tour_len2.sum(1).detach()

    obj = w1*obj1 + w2*obj2
    return obj, obj1, obj2



def render(static, tour_indices, save_path):
    """Plots the found tours."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
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