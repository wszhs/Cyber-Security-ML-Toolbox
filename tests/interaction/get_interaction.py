'''
Author: your name
Date: 2021-06-24 20:18:28
LastEditTime: 2021-07-23 14:11:45
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/image/interaction_loss.py
'''
import numpy as np
import torch
import torch.nn as nn
import copy
import random
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


class InteractionLoss(nn.Module):

    def __init__(self, target=None, label=None):
        super(InteractionLoss, self).__init__()
        assert (target is not None) and (label is not None)
        self.target = target
        self.label = label

    def logits_interaction(self, outputs, leave_one_outputs,
                           only_add_one_outputs, zero_outputs):
        complete_score = outputs[:, self.target] - outputs[:, self.label]
        leave_one_out_score = (
            leave_one_outputs[:, self.target] -
            leave_one_outputs[:, self.label])
        only_add_one_score = (
            only_add_one_outputs[:, self.target] -
            only_add_one_outputs[:, self.label])
        zero_score = (
            zero_outputs[:, self.target] - zero_outputs[:, self.label])

        average_pairwise_interaction = (complete_score - leave_one_out_score -
                                        only_add_one_score +
                                        zero_score).mean()
        return average_pairwise_interaction

    def forward(self, outputs, leave_one_outputs, only_add_one_outputs,
                zero_outputs):
        return self.logits_interaction(outputs, leave_one_outputs,
                                       only_add_one_outputs, zero_outputs)


def sample_grids(sample_grid_num,
                 grid_scale,
                 sample_times):
    
    sample = []
    for _ in range(sample_times):
        # total = 135
        # li = [i for i in range(total)]
        # ids=random.sample(li,sample_grid_num)
        ids = np.random.randint(0, grid_scale, size=sample_grid_num)
        sample.append(ids)
    return sample


def sample_for_interaction(delta,
                           sample_grid_num,
                           grid_scale,
                           times):
    samples = sample_grids(
        sample_grid_num=sample_grid_num,
        grid_scale=grid_scale,
        sample_times=times)
    only_add_one_mask = torch.zeros_like(delta).repeat(times,1)
    # print(only_add_one_mask.shape)
    for i in range(times):
        grids = samples[i]
        for grid in grids:
            only_add_one_mask[i,grid] = 1
    leave_one_mask = 1 - only_add_one_mask
    # plt.matshow(only_add_one_mask)
    # plt.clim(0, 1)
    # plt.show()
    only_add_one_perturbation = delta * only_add_one_mask
    leave_one_out_perturbation = delta * leave_one_mask

    return only_add_one_perturbation, leave_one_out_perturbation


def get_features(
    model,
    x,
    perturbation,
    leave_one_out_perturbation,
    only_add_one_perturbation,
):

    outputs = model(x + perturbation)
    leave_one_outputs = model(x + leave_one_out_perturbation)
    only_add_one_outputs = model(x + only_add_one_perturbation)
    zero_outputs = model(x)

    return (outputs, leave_one_outputs, only_add_one_outputs, zero_outputs)

def get_average_pairwise_interaction(model,x,y,delta,sample_grid_num,grid_scale,times):
    only_add_one_perturbation, leave_one_out_perturbation = sample_for_interaction(delta,sample_grid_num=sample_grid_num,grid_scale=grid_scale,times=times)
    (outputs, leave_one_outputs, only_add_one_outputs,zero_outputs) = get_features(model, x, delta,leave_one_out_perturbation,only_add_one_perturbation)
    label=y.item()
    # outputs_c = copy.deepcopy(outputs.detach())
    # outputs_c[:, label] = -np.inf
    other_max = 0
    interaction_loss = InteractionLoss(target=other_max, label=label)
    average_pairwise_interaction = interaction_loss(outputs, leave_one_outputs, only_add_one_outputs,zero_outputs)
    return average_pairwise_interaction