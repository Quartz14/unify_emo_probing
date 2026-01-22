import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from tqdm.auto import tqdm
import os 
import matplotlib


%load_ext autoreload
%autoreload 2

model_names = ['Llama3.1_8B','gemma-2-9b-it']
layers_counts = [32, 42]


indices = np.argsort(layers_counts)

model_names = [model_names[i] for i in indices]
layers_counts = [layers_counts[i] for i in indices]

print(model_names, layers_counts)

probe_location_to_formal_name = {3: 'Attention Output', 6: 'MLP Output', 7: 'Hidden State'}

num_exps = 200
locs_to_probe = [3, 6, 7]
tokens = [-1] 
span_patch = 5
max_layer_count = max(layers_counts) - span_patch + 1
values_patch = torch.zeros((len(locs_to_probe), len(model_names), max_layer_count))
values_patch = values_patch.fill_(np.nan)

task_type = 'llm'
prompt_version = 'P2'


def plot_heatmap(vals, titles, xticks, xtick_labels, fontsize, yticks, ytick_labels, yticks_rotation, xticks_rotation,
                 suptitle = None, subtitle='Probe at', vmax=None, vmin=None,
                 y_axis_label='Tokens', x_axis_label='Layers',
                 cmap='magma', cmap_label='Accuracy', cmap_shrink=1.0, cmap_aspect=10, cmap_fraction=0.03, cmap_pad=0.02,
                 cbar_yticks=None, cbar_ytick_labels=None,
                 figsize=(20, 12), save_path = ''):
    
    if vmax is None:
        vmax = vals.max().item()
    if vmin is None:
        vmin = vals.min().item()
        
    if cbar_yticks is None or cbar_ytick_labels is None:
        # Format to 2 decimal places
        vmax_str = f'{vmax:.2f}'
        vmin_str = f'{vmin:.2f}'
        cbar_yticks = [vmin, vmax]
        cbar_ytick_labels = [vmin_str, vmax_str]

    fig, axes = plt.subplots(1, vals.shape[0], figsize=figsize, constrained_layout=True)
    
    if vals.shape[0] == 1:
        axes = [axes]
    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize)

    norm = plt.Normalize(vmax, vmin)

    for i in range(len(vals)):
        cbar = False
        sns.heatmap(vals[i], ax=axes[i], cmap=cmap, vmin=vmin, vmax=vmax, cbar=cbar, square=True)
        
        axes[i].set_title(titles[i], fontsize=fontsize)
        axes[i].set_xlabel(x_axis_label, fontsize=fontsize)
        if i == 0:
            axes[i].set_ylabel(y_axis_label, fontsize=fontsize)

        axes[i].set_xticks(xticks, xtick_labels, fontsize = fontsize / 1.2, rotation=xticks_rotation)
        if i == 0:
            axes[i].set_yticks(yticks, ytick_labels, fontsize = fontsize, rotation=yticks_rotation)
        else:
            axes[i].set_yticks([])
        axes[i].tick_params(axis='both', which='both', length=0)


    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='vertical',
                        shrink=cmap_shrink, aspect=cmap_aspect, fraction=cmap_fraction, pad=cmap_pad)
    cbar.set_label(cmap_label, fontsize=fontsize)
    cbar.outline.set_color('white')

    # turn off cbar ticks
    # cbar.ax.set_yticklabels(['0', '1'])
    cbar.ax.set_yticks(cbar_yticks, cbar_ytick_labels, fontsize=fontsize / 1.2)
    cbar.ax.tick_params(labelsize=fontsize / 1.2, axis='both', which='both', length=0)
    if save_path != '':
        plt.savefig(f'plots_expt/{save_path}.pdf', bbox_inches='tight', dpi=500)
    plt.show()
                   


for i, (model_name, layer_count) in enumerate(zip(model_names, layers_counts)):
    layers_centers = list(range(span_patch // 2, layer_count - span_patch // 2))
    print(layers_centers)
    v = process_activation_patching(model_name, num_exps, locs_to_probe, tokens, layers_centers, span_patch, task_type, prompt_version)
    v = v[:, tokens.index(-1)][..., 0]
    
    print(values_patch.shape, v.shape)
    
    for j, _ in enumerate(locs_to_probe):
        for k, _ in enumerate(layers_centers):
            values_patch[j, i, k] = v[j, k]
    break



# for llama-3.1-8B-inst
layers_centers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

# layers_centers = list(range(span_patch // 2, layers_counts[1] - span_patch // 2)) # For gemma
print(layers_centers)
model_name = model_names[0]
# model_name = model_names[1]

values_patch = torch.zeros(3,5,len(layers_centers))#torch.zeros((len(locs_to_probe), len(model_names), max_layer_count))
values_patch = values_patch.fill_(np.nan)

datasets = ['crowd', 'crowd-h', 'dialogue', 'llm', 'twitter']
for i,task_type in enumerate(['crowd', 'crowd-h', 'dialogue', 'llm', 'twitter']):
    if(task_type == 'dialogue'):
        prompt_version = 'P2-d'
        # prompt_version = 'P2d'
    else:
        prompt_version = 'P2'
    v = process_activation_patching(model_name, num_exps, locs_to_probe, tokens, layers_centers, span_patch, task_type, prompt_version)
    v = v[:, tokens.index(-1)][..., 0]

    for j, _ in enumerate(locs_to_probe):
        for k, _ in enumerate(layers_centers):
            values_patch[j, i, k] = v[j, k]


FONT_SIZE = 16
datasets = ['crowd', 'crowd-h', 'dialogue', 'llm', 'twitter']

layers_centers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

# LLAMA
plot_heatmap(values_patch, suptitle = None, #f'Patching (with span {span_patch})', 
             titles = [f'{probe_location_to_formal_name[loc]}' for loc in locs_to_probe], fontsize = FONT_SIZE,
             xticks = np.arange(len(layers_centers)) + 0.5, xtick_labels = (torch.tensor(layers_centers) + 1).numpy(),
             yticks = np.arange(len(datasets)) + 0.5, ytick_labels = datasets,
             yticks_rotation = 0, xticks_rotation = 90,
             vmax=1.0, vmin=0.0,
             x_axis_label='Layers - Llama-3.1-8B-inst', y_axis_label='Datasets',
             cmap='rocket', cmap_label='Patching Success', cmap_shrink=0.8, cmap_aspect=7.0, cmap_fraction=0.01, cmap_pad=0.01,
             cbar_yticks=[-0.1, 1.0], cbar_ytick_labels=[0, 1],
             figsize=(25, 17), save_path = f'Llama_8B_emotion_patch_heatmap')

# GEMMA
plot_heatmap(values_patch, suptitle = None, #f'Patching (with span {span_patch})', 
             titles = [f'{probe_location_to_formal_name[loc]}' for loc in locs_to_probe], fontsize = FONT_SIZE,
             xticks = np.arange(len(layers_centers)) + 0.5, xtick_labels = (torch.tensor(layers_centers) + 1).numpy(),
             yticks = np.arange(len(datasets)) + 0.5, ytick_labels = datasets,
             yticks_rotation = 0, xticks_rotation = 90,
             vmax=1.0, vmin=0.0,
             x_axis_label='Layers - Gemma-2-9B-inst', y_axis_label='Datasets',
             cmap='rocket', cmap_label='Patching Success', cmap_shrink=0.8, cmap_aspect=7.0, cmap_fraction=0.01, cmap_pad=0.01,
             cbar_yticks=[-0.1, 1.0], cbar_ytick_labels=[0, 1],
             figsize=(25,17), save_path = f'Gemma_9B_emotion_patch_heatmap')




