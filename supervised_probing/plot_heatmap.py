import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
import torch
import json

f = open("plot_res_all.json")
df = json.load(f)

model_to_layers = {"Llama3.2_1B":16, "Llama3.2_3B":28, 'Llama3.1_8B':32,'gemma-2-2b-it':26, 'gemma-2-9b-it':42 }
loc_to_id = {'attention':'3', 'mlp':'6', 'residual':'7'}
models = []
prompts = []
datasets = []
layers = []
residuals = []
attentions = [] 
mlps = []
for elem_index in list(range(len(df))):
    model= df[elem_index]['model']
    prompt = df[elem_index]['prompt']
    dataset = df[elem_index]['dataset']
    layers = model_to_layers[model]
    residual = []
    attention = []
    mlp = []
    
    for l in list(range(layers)):
        l = str(l)
        # print(df[elem_index])
        residual.append(df[elem_index][l][loc_to_id['residual']][0])
        attention.append(df[elem_index][l][loc_to_id['attention']][0])
        mlp.append(df[elem_index][l][loc_to_id['mlp']][0])
    
    residuals.append(residual)
    attentions.append(attention)
    mlps.append(mlp)
    models.append(model)
    prompts.append(prompt)
    datasets.append(dataset)


df_consolidated = pd.DataFrame()
df_consolidated['model'] = models
df_consolidated['dataset'] = datasets
df_consolidated['prompt'] = prompts
df_consolidated['residual'] = residuals
df_consolidated['attention'] = attentions
df_consolidated['mlp'] = mlps

# df_consolidated.head() , len(df)
# df_consolidated = df_consolidated.drop_duplicates(subset=['model', 'dataset', 'prompt'])




for model in ['Llama3.2_3B', 'Llama3.2_1B', 'gemma-2-2b-it', 'gemma-2-9b-it','Llama3.1_8B']:
# for model in ['gemma-2-9b-it']:
    df_sub = df_consolidated[df_consolidated['model']==model]
    

    df_sub = df_sub.sort_values(by=['dataset', 'prompt'], ascending=[True, True])
    y_labels = list(df_sub['prompt'])
    group_labels = list(df_sub['dataset'])
    element_counts = Counter(group_labels)
    group_labels = list(element_counts.items())
    layers = model_to_layers[model]

    for loc in ['attention', 'mlp']:
    # for loc in ['residual']:
        acc = torch.zeros(len(y_labels),layers)
        values = list(df_sub[loc])
        # for i in range(len(y_labels)):
        for i in range(len(y_labels)):
            acc[i] = torch.tensor(values[i])
        
        print(f"------------------Model: {model}, Location: {loc}---------------------")
        save_file = model+"_"+loc
        # plot(acc,y_labels, group_labels, f'{model}_{loc}')
        plt.figure(figsize=(16,4), dpi=100)
        ax = sns.heatmap(acc, annot=True, fmt='.2f', annot_kws={"fontsize":7.5},
                    cbar_kws={"shrink": 0.8, "pad":0.01})

        # Set fine y-axis labels
        ax.set_yticks(np.arange(len(y_labels)) + 0.5)
        ax.set_yticklabels(y_labels, fontsize=7.5, rotation=0)

        # Add vertical group labels manually (same side as y-axis)
        y_start = 0
        for i, (label, size) in enumerate(group_labels):
            center = y_start + size/2
            ax.text(
                -1.6, center, label,
                va='center', ha='center', fontsize=8, fontweight='bold',
                rotation=90   # <-- vertical orientation
            )
            # Draw a horizontal line *after* this group (except the last one)
            if i < len(group_labels) - 1:
                ax.hlines(y_start + size, *ax.get_xlim(),
                    colors="black", linewidth=0.8, linestyles="--")
            y_start += size

        # Adjust x-axis and colorbar font size
        ax.tick_params(axis='x', labelsize=7.5)
        ax.tick_params(axis='y', length=0)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5)

        plt.xticks(rotation=0, ha="right")
        plt.tight_layout()
        plt.savefig(f"plots/{save_file}.pdf", dpi=500, bbox_inches='tight')
        plt.show()
        # break

    
    # print(model,len(df_sub))
