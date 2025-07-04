import os
import random
import logging
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import manifold
from umap import UMAP
from .dataloader import get_dataloader, get_dict_dataloader


def visualize_plm(model, dataset, triggers, save_dir, image_name, dim_reduce):
    clean_dataset = dataset['train-clean']
    dataset = sum([v for k, v in dataset.items() if k.split("-")[0]=='train'], [])
    dataloader = get_dataloader(dataset, batch_size=16, drop_last=False)
    data_iterator = tqdm(dataloader, desc="Evaluating")

    # get data cls embeds
    model.eval()
    all_hidden_states = []
    all_labels = []
    for batch in data_iterator:
        inputs, _, labels  = model.process(batch)
        with torch.no_grad():
            outputs = model(inputs)
            if hasattr(outputs, 'last_hidden_state'):
                cls_embeds = outputs.last_hidden_state[:,0,:]
            else:
                cls_embeds = outputs.hidden_states[-1][:,0,:]   # for MaskedLanguageModel
        all_hidden_states.extend(cls_embeds.detach().cpu().tolist())
        all_labels.extend(labels.view(-1).detach().cpu().tolist())

    # get embeds with 2-dims
    if dim_reduce == 'umap':
        data_embeds = np.array(dimension_reduction_umap(all_hidden_states))
    if dim_reduce == 'tsne': 
        data_embeds = np.array(dimension_reduction_tsne(all_hidden_states))

    # set color and alpha
    
    color_map = ['#FFD700', '#11FFFF', '#D633FF', '#8FFF44', '#4469FF', '#FF5577', '#8E99C6', '#31A4F0', '#7011FF', '#D0E86C', '#A239E9', '#E0E064', '#E9A239', '#AD8585', '#D4D4D4', '#9FB57E']
    alpha = (0.2,)
    all_colors = [colors.to_rgba(color_map[l])[:-1] + alpha for l in all_labels]
    all_edge_colors = [color_map[l] for l in all_labels]

    # plot
    sns.set()
    fig, ax = plt.subplots()
    ax.scatter(data_embeds[:, 0], data_embeds[:, 1], marker='o', s=8, c=all_colors, linewidths=.3, edgecolor=all_edge_colors)
    ax.tick_params(pad=0.1, labelsize=10)

    triggers.insert(0, 'clean')
    for i, trigger in enumerate(triggers):
        if i == 0:
            plt.scatter([], [], c=color_map[i], s=8, label=trigger)
        else:
            plt.scatter([], [], c=color_map[i], s=8, label='poison-'+trigger)
    plt.legend(facecolor='white')

    plt.savefig(os.path.join(save_dir, image_name+'.pdf'))
    plt.savefig(os.path.join(save_dir, image_name+'.png'))
    plt.close()


def visualize_dsm(model, clean_dataset, poisoner, num_labels, save_dir, image_name, dim_reduce):
    dataset = defaultdict(list)
    dataset["test-clean"] = clean_dataset["test"]
    dataset.update(poisoner.poison_all_test_dataset(clean_dataset["test"]))   # poisoning all samples, not just samples with non-target label 
    dataloader = get_dict_dataloader(dataset, batch_size=16)

    # get data cls embeds
    model.eval()
    all_nums, all_hidden_states, all_preds = [], [], []
    for i, key in enumerate(dataloader.keys()):
        all_nums.append(len(dataset[key]))
        data_iterator = tqdm(dataloader[key], desc="Evaluating")
        for batch in data_iterator:
            inputs, _  = model.process(batch)
            with torch.no_grad():
                outputs = model(inputs)
                cls_embeds = outputs.hidden_states[-1][:,0,:]
                preds = torch.argmax(outputs.logits, dim=-1)
                all_hidden_states.extend(cls_embeds.detach().cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

    # get embeds with 2-dims
    if dim_reduce == 'umap':
        data_embeds = np.array(dimension_reduction_umap(all_hidden_states))
    if dim_reduce == 'tsne': 
        data_embeds = np.array(dimension_reduction_tsne(all_hidden_states))

    # set color and alpha
    color_map = ['#FF5577', '#1E90FF']
    alpha = (0.2,)
    marker_map = ['o', 'v', 's', 'p', 'h', 'D']
    all_colors, all_edge_colors, all_markers = [], [], []

    start = 0
    for i, num in enumerate(all_nums):
        all_colors.extend([colors.to_rgba(color_map[l])[:-1] + alpha for l in all_preds[start : start + num]])
        all_edge_colors.extend([color_map[l] for l in all_preds[start : start + num]])
        start += num
        
    # plot
    sns.set()
    fig, ax = plt.subplots()
    start = 0
    for i, num in enumerate(all_nums):
        ax.scatter(data_embeds[start : start + num][:, 0], 
                   data_embeds[start : start + num][:, 1], 
                   marker=marker_map[i], s=12, linewidths=.3,
                   c=all_colors[start : start + num], 
                   edgecolor=all_edge_colors[start : start + num])
        start += num
    ax.tick_params(pad=0.1, labelsize=10)

    triggers = poisoner.get_triggers()
    triggers.insert(0, 'clean')
    lengend_labels = []
    for i, trigger in enumerate(triggers):
        lengend_labels.append([])
        for j in range(num_labels):
            if i == 0:
                lengend_labels[i].append(trigger+'-label-'+str(j))
            else:
                lengend_labels[i].append('poison-'+trigger+'-label-'+str(j))
    
    for i, trigger in enumerate(triggers):
        for j in range(num_labels):
            plt.scatter([], [], marker=marker_map[i], c=colors.to_rgba(color_map[j])[:-1] + alpha, edgecolor=color_map[j], s=12, label=lengend_labels[i][j])
    plt.legend(facecolor='white')

    plt.savefig(os.path.join(save_dir, image_name+'.pdf'))
    plt.savefig(os.path.join(save_dir, image_name+'.png'))
    plt.close()


def dimension_reduction_umap(hidden_states):
    pca = PCA(n_components=40, random_state=42)
    umap = UMAP(n_neighbors=120, min_dist=0.7, n_components=2, random_state=42, transform_seed=42)
    embedding_pca = pca.fit_transform(hidden_states)
    embeddings = umap.fit(embedding_pca).embedding_
    return embeddings


def dimension_reduction_tsne(hidden_states):
    embeddings = manifold.TSNE(n_components=2, init='pca', perplexity=30, random_state=42).fit_transform(np.array(hidden_states))
    return embeddings



