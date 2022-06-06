'''
Author: your name
Date: 2021-07-04 14:05:25
LastEditTime: 2021-07-04 17:24:27
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/image/get_interaction.py
'''
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import copy

def sample_grids(sample_grid_num,
                 grid_scale,
                 img_size,
                 sample_times,
                 con_i,
                 con_j):
    grid_size = img_size // grid_scale
    samples_two = []
    samples_i = []
    samples_j = []
    samples_o = []
    
    for _ in range(sample_times):
        grids_two = []
        grids_i = []
        grids_j = []
        grids_o=[]
        ids = np.random.randint(0, grid_scale**2, size=sample_grid_num)
        ids=np.append(ids,con_i)
        ids=np.append(ids,con_j)
        rows, cols = ids // grid_scale, ids % grid_scale
        for r, c,id in zip(rows, cols,ids):
            grid_range = (slice(r * grid_size, (r + 1) * grid_size),
                          slice(c * grid_size, (c + 1) * grid_size))
            if id==con_i:
                grids_two.append(grid_range)
                grids_i.append(grid_range)
            elif id==con_j:
                grids_two.append(grid_range)
                grids_j.append(grid_range)
            else:
                grids_two.append(grid_range)
                grids_i.append(grid_range)
                grids_j.append(grid_range)
                grids_o.append(grid_range)
        samples_two.append(grids_two)
        samples_i.append(grids_i)
        samples_j.append(grids_j)
        samples_o.append(grids_o)
    return samples_two,samples_i,samples_j,samples_o

def sample_for_interaction_zhs(delta,con_i,con_j,
                           sample_grid_num=100,
                           grid_scale=14,
                           img_size=28,
                           times=50):
    samples_two,samples_i,samples_j,samples_o = sample_grids(
        sample_grid_num=sample_grid_num,
        grid_scale=grid_scale,
        img_size=img_size,
        sample_times=times,
        con_i=con_i,
        con_j=con_j)
    two_mask = torch.zeros_like(delta).repeat(times, 1, 1, 1)
    i_mask=torch.zeros_like(delta).repeat(times, 1, 1, 1)
    j_mask=torch.zeros_like(delta).repeat(times, 1, 1, 1)
    o_mask=torch.zeros_like(delta).repeat(times, 1, 1, 1)
    for i in range(times):
        grids_two = samples_two[i]
        grids_i = samples_i[i]
        grids_j = samples_j[i]
        grids_o=samples_o[i]
        for grid_two in grids_two:
            two_mask[i:i + 1, :, grid_two[0], grid_two[1]] = 1
        for grid_i in grids_i:
            i_mask[i:i + 1, :, grid_i[0], grid_i[1]] = 1
        for grid_j in grids_j:
            j_mask[i:i + 1, :, grid_j[0], grid_j[1]] = 1
        for grid_o in grids_o:
            o_mask[i:i + 1, :, grid_o[0], grid_o[1]] = 1

    two_perturbation = delta * two_mask
    i_perturbation = delta * i_mask
    j_perturbation = delta * j_mask
    o_perturbation = delta * o_mask
    return two_perturbation,i_perturbation,j_perturbation,o_perturbation

def get_score_zhs(
    model,
    x,
    perturbation,
    two_perturbation,
    i_perturbation,
    j_perturbation,
    o_perturbation
):
    outputs = model(x + perturbation)
    two_outputs = model(x + two_perturbation)
    i_outputs = model(x + i_perturbation)
    j_outputs = model(x + j_perturbation)
    o_outputs = model(x + o_perturbation)

    return (outputs,two_outputs,i_outputs,j_outputs,o_outputs)


def logits_interaction_zhs(target,label,two_outputs,i_outputs,j_outputs,o_outputs):
    two_score = two_outputs[:, target] - two_outputs[:, label]
    i_score = i_outputs[:, target] - i_outputs[:, label]
    j_score = j_outputs[:, target] - j_outputs[:, label]
    o_score = o_outputs[:, target] - o_outputs[:, label]

    average_pairwise_interaction = (two_score - i_score -j_score +o_score).mean()
    return average_pairwise_interaction

def get_average_pairwise_interaction(model,x,y,delta,con_i,con_j):
    two_perturbation,i_perturbation,j_perturbation,o_perturbation=sample_for_interaction_zhs(delta=delta,con_i=con_i,con_j=con_j)
    outputs,two_outputs,i_outputs,j_outputs,o_outputs=get_score_zhs(model,x,delta,two_perturbation,i_perturbation,j_perturbation,o_perturbation)
    label=y.item()
    outputs_c = copy.deepcopy(outputs.detach())
    outputs_c[:, label] = -np.inf
    other_max = outputs_c.max(1)[1].item()
    average_pairwise_interaction=logits_interaction_zhs(other_max,label,two_outputs,i_outputs,j_outputs,o_outputs)
    print(average_pairwise_interaction)

