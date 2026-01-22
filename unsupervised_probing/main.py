from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns

from repe import repe_pipeline_registry
repe_pipeline_registry()

from utils import primary_emotions_concept_dataset, primary_emotions_concept_dataset_crowd, primary_emotions_concept_dataset_twitter, primary_emotions_concept_dataset_dialogue

import os

# To make only GPU device 0 visible
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--prompt", type=str, default="P2")
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()


model_name_or_path = args.model_path

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", token=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, , use_fast=True, padding_side="left", legacy=False, token=True)
tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

n_layers = model.config.num_hidden_layers


rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = 'pca'
rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

# For LLM, crowd
emotions_llm_crowd = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
# For Dialogue, 
emotions_dialogue = ["joy", "sadness", "anger", "fear", "disgust", "surprise"]
# For Twitter
emotions_twitter = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def get_rep_readers(data, emotions):
    emotion_rep_readers = {}
    if 'joy' in data:
        data['happiness'] = data.pop('joy')
    emotions_to_process = ['happiness' if e == 'joy' else e for e in emotions]

    for emotion in tqdm(emotions_to_process):

        train_data = data[emotion]['train']
    
        rep_reader = rep_reading_pipeline.get_directions(
            train_data['data'], 
            rep_token=rep_token, 
            hidden_layers=hidden_layers, 
            n_difference=n_difference, 
            train_labels=train_data['labels'], 
            direction_method=direction_method,
        )

        emotion_rep_readers[emotion] = rep_reader
    return emotion_rep_readers

def get_test_rep(data, emotions, emotion_rep_readers):
    emotion_H_tests = {}
    if 'joy' in data:
        data['happiness'] = data.pop('joy')
    emotions_to_process = ['happiness' if e == 'joy' else e for e in emotions]

    for emotion in tqdm(emotions_to_process):
        
        test_data = data[emotion]['test']
        
        if(emotion=='love' and 'love' not in list(emotion_rep_readers.keys())):
            rep_reader = emotion_rep_readers['happiness']
        elif(emotion=='disgust' and 'disgust' not in list(emotion_rep_readers.keys())):
            # emotion_H_tests[emotion] = np.zeros(H_tests.shape())
            continue
        else:
            rep_reader = emotion_rep_readers[emotion]

        H_tests = rep_reading_pipeline(
            test_data['data'], 
            rep_token=rep_token, 
            hidden_layers=hidden_layers, 
            rep_reader=rep_reader,
            batch_size=int(args.batch_size))

        emotion_H_tests[emotion] = H_tests
    return emotion_H_tests


def get_acc(data , reader, emotions, plot, fn):
    results = {layer: {} for layer in hidden_layers}
    if 'joy' in data:
        data['happiness'] = data.pop('joy')
    emotions_to_process = ['happiness' if e == 'joy' else e for e in emotions]

    for layer in hidden_layers:
        for idx, emotion in enumerate(emotions_to_process):
            H_test = [H[layer] for H in data[emotion]] 
            H_test = [H_test[i:i+2] for i in range(0, len(H_test), 2)]
            
            if(emotion == 'disgust' and 'disgust' not in list(reader.keys())):
                results[layer][emotion] = 0
                continue

            elif(emotion == 'love' and 'love' not in list(reader.keys())):
                sign = reader['happiness'].direction_signs[layer]
            else:
                sign = reader[emotion].direction_signs[layer]
            
            eval_func = min if sign == -1 else max
            
            cors = np.mean([eval_func(H) == H[0] for H in H_test])
            
            results[layer][emotion] = cors



    if(plot):
        for emotion in emotions_to_process:
        # x = list(results.keys())
            x = [model.config.num_hidden_layers+i for i in list(results.keys())]
            y = [results[layer][emotion] for layer in results]

            plt.plot(x, y, label=emotion)
        plt.title("Emotions Accuracy")
        plt.xlabel("Layer (Llama-3.1-8B-Instruct)")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.4)

        file_path = f'plots/pca/{fn}.pdf'
        plt.savefig(
            file_path,
            dpi=500,                # Set the resolution to 500 dots per inch
            format='pdf',            # Explicitly set the format to PDF
            bbox_inches='tight'      # Crop the saved figure to a tight bounding box
        )
        plt.cla()

    return results

#Calculate for all data

def dict_to_df(dataset_dict, dataset_name):
    rows = []
    for layer, emo_scores in dataset_dict.items():
        for emo, score in emo_scores.items():
            rows.append({"layer": int(layer), "emotion": emo, "score": score, "dataset": dataset_name})
    return pd.DataFrame(rows)

def extract_max_scores(all_data):
    rows = []
    
    for dataset_name, layer_dict in all_data.items():
        # Collect per emotion across layers
        emotion_scores = {}
        for layer, emo_scores in layer_dict.items():
            layer = int(layer)
            for emo, score in emo_scores.items():
                if emo not in emotion_scores:
                    emotion_scores[emo] = []
                emotion_scores[emo].append((layer, score))
        
        # Find max and earliest layer
        for emo, values in emotion_scores.items():
            max_score = max(s for _, s in values)
            # Find earliest layer with this max
            earliest_layer = min(l for l, s in values if s == max_score)
            rows.append({
                "dataset": dataset_name,
                "emotion": emo,
                "max_acc_score": max_score,
                "earliest_layer": earliest_layer
            })
    
    return pd.DataFrame(rows)

def compute_similarities_with_summary(emotion_rep_readers, emotion_rep_readers_P2, hidden_layers):
    similarities = {}
    summary_rows = []

    for emotion in emotion_rep_readers_P2.keys():
        simi = []
        for layer in hidden_layers:
            # If missing in base, fill zero similarity
            if emotion not in emotion_rep_readers:
                if('emotion' == 'love'):
                    v1 = emotion_rep_readers['happiness'].directions[layer]
                    v1_sign = emotion_rep_readers['happiness'].direction_signs[layer]
                else:
                    simi.append(0)
                    continue
            else:
                v1 = emotion_rep_readers[emotion].directions[layer]
                v1_sign = emotion_rep_readers[emotion].direction_signs[layer]
            v1 = v1 * v1_sign

            v2 = emotion_rep_readers_P2[emotion].directions[layer]
            v2_sign = emotion_rep_readers_P2[emotion].direction_signs[layer]
            v2 = v2 * v2_sign

            v1 = v1.flatten()
            v2 = v2.flatten()

            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 > 0 and norm_v2 > 0:
                similarity = dot_product / (norm_v1 * norm_v2)
                simi.append(similarity)
            else:
                simi.append(0)
        similarities[emotion] = simi

        # Find max score and earliest layer
        max_score = max(simi) if simi else 0
        if simi:
            earliest_layer = hidden_layers[simi.index(max_score)]
        else:
            earliest_layer = None

        summary_rows.append({
            "emotion": emotion,
            "max_similarity": max_score,
            "earliest_layer": earliest_layer
        })

    return similarities, pd.DataFrame(summary_rows)


data_llm = primary_emotions_concept_dataset('no_emotion',"dataset/emotions")
rep_readers = get_rep_readers(data_llm, emotions_llm_crowd)
torch.save(rep_readers, f'results/PCA_reading_vecs/{args.prompt}_twitter.pt')
rep_readers = torch.load(f'results/PCA_reading_vecs/{args.prompt}.pt', weights_only=False)

data_twitter = primary_emotions_concept_dataset_twitter()
rep_readers_twitter = get_rep_readers(data_twitter, emotions_twitter)
torch.save(rep_readers_twitter, f'results/PCA_reading_vecs/{args.prompt}_twitter.pt')
rep_readers_twitter = torch.load(f'results/PCA_reading_vecs/{args.prompt}_twitter.pt', weights_only=False)

data_dialogue = primary_emotions_concept_dataset_dialogue()
rep_readers_dialogue = get_rep_readers(data_dialogue, emotions_dialogue)
torch.save(rep_readers_dialogue, f'results/PCA_reading_vecs/{args.prompt}_dialogue.pt') # last and full
rep_readers_dialogue = torch.load(f'results/PCA_reading_vecs/{args.prompt}_dialogue.pt', weights_only=False)



data_crowd = primary_emotions_concept_dataset_crowd(hidden=False)
rep_readers_crowd = get_rep_readers(data_crowd, emotions_llm_crowd)
torch.save(rep_readers_crowd , f'results/PCA_reading_vecs/{args.prompt}_crowd.pt')
rep_readers_crowd = torch.load(f'results/PCA_reading_vecs/{args.prompt}_crowd.pt', weights_only=False)



data_crowd_hidden = primary_emotions_concept_dataset_crowd(hidden=True)
rep_readers_crowd_hidden = get_rep_readers(data_crowd_hidden, emotions_llm_crowd)
torch.save(rep_readers_crowd_hidden, f'results/PCA_reading_vecs/{args.prompt}_crowd_hidden.pt')
rep_readers_crowd_hidden = torch.load(f'results/PCA_reading_vecs/{args.prompt}_crowd_hidden.pt', weights_only=False)


llmP2_cosine = {}

all_P2 = {
    "data_crowd": rep_readers_crowd,
    "data_crowd_hidden": rep_readers_crowd_hidden,
    "data_twitter": rep_readers_twitter,
    "data_dialogue": rep_readers_dialogue
}

all_summaries = []

for name, P2 in all_P2.items():
    similarities, summary_df = compute_similarities_with_summary(
        rep_readers, P2, hidden_layers
    )
    summary_df["comparison"] = name
    all_summaries.append(summary_df)
    llmP2_cosine[name] = similarities

final_summary = pd.concat(all_summaries, ignore_index=True)

# Save summary to CSV
final_summary.to_csv("results/LLMP2_similarity_summary.csv", index=False)
print(final_summary.head())

layers_nonneg = [n_layers+i for i in hidden_layers]

all_emotions = set(["happiness", "sadness", "anger", "fear", "disgust", "surprise"])#, "love", "joy"])
emotion_to_color = {"happiness": "#1f77b4","sadness":"#ff7f0e" , "anger": "#2ca02c", "fear": "#d62728", "disgust": "#9467bd", "surprise": "#8c564b" }


fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
for idx, kv in enumerate(llmP2_cosine.items()):
    similarities = kv[1]
    data_name = kv[0]

    ax = axes[idx]
    for emotion, simi in similarities.items():
        if(emotion in all_emotions):
            ax.plot(layers_nonneg, simi, marker='.', linestyle='-',
                label=emotion, color=emotion_to_color[emotion])
    ax.set_title(f"Comparison {data_name}-data_LLM vectors")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlabel("Layer")
    if idx == 0:
        ax.set_ylabel("Cosine similarity")

# --- Single shared legend outside ---
handles, labels = [], []
for emo in all_emotions:
    handles.append(plt.Line2D([0], [0], color=emotion_to_color[emo], lw=2))
    labels.append(emo)

fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=15)
plt.tight_layout(rect=[0, 0, 0.8, 0])
file_path = f'results/plots/final_plots/all_{args.prompt}_cosin_sim_vectors.pdf'
plt.savefig(
            file_path,
            dpi=500,                # Set the resolution to 500 dots per inch
            format='pdf',            # Explicitly set the format to PDF
            bbox_inches='tight'      # Crop the saved figure to a tight bounding box
        )
plt.cla()


print(">>> Cosine similarity calculations done!!!") 

#Apply to all datasets!
all_data = {}

emotion_H_tests_dialogue = get_test_rep(data_dialogue, emotions_dialogue, rep_readers_dialogue)
res = get_acc(emotion_H_tests_dialogue, rep_readers_dialogue, emotions_dialogue, True, 'dialogue_self')
all_data['dialogue'] = res

emotion_H_tests_twitter = get_test_rep(data_twitter, emotions_twitter, rep_readers_twitter)
res = get_acc(emotion_H_tests_twitter, rep_readers_twitter, emotions_twitter, True, 'twitter_self')
all_data['twitter'] = res

emotion_H_tests = get_test_rep(data_llm, emotions_llm_crowd, rep_readers)
res = get_acc(emotion_H_tests, rep_readers, emotions_llm_crowd, True, 'llm_self')
all_data['llm'] = res


emotion_H_tests_crowd = get_test_rep(data_crowd, emotions_llm_crowd, rep_readers_crowd)
res = get_acc(emotion_H_tests_crowd, rep_readers_crowd, emotions_llm_crowd, True, 'crowd_self')
all_data['crowd'] = res

emotion_H_tests_crowd_hidden = get_test_rep(data_crowd_hidden, emotions_llm_crowd, rep_readers_crowd_hidden)
res = get_acc(emotion_H_tests_crowd_hidden, rep_readers_crowd_hidden, emotions_llm_crowd, True, 'crowd_hidden_self')
all_data['crowd_hidden'] = res


print(">>> All self plots saved!")

df = extract_max_scores(all_data)

# Save as CSV
df.to_csv(f"results/PCA_self_layer_scores{args.prompt}.csv", index=False)
torch.save(all_data,f'results/PCA_reading_vecs/results_all_data_self{args.prompt}.pt')

print(df.head())
print(">>> All self data saved!")



all_data = {}
emotion_H_tests_crowd = get_test_rep(data_crowd, emotions_llm_crowd,rep_readers)
res = get_acc(emotion_H_tests_crowd, rep_readers, emotions_llm_crowd, True, f'crowd_llm{args.prompt}')
all_data['crowd'] = res

emotion_H_tests_crowd_hidden = get_test_rep(data_crowd_hidden, emotions_llm_crowd, rep_readers)
res = get_acc(emotion_H_tests_crowd_hidden, rep_readers, emotions_llm_crowd, True, f'crowd_hidden_llm{args.prompt}')
all_data['crowd_hidden'] = res

emotion_H_tests_twitter = get_test_rep(data_twitter, emotions_twitter, rep_readers)
res = get_acc(emotion_H_tests_twitter, rep_readers, emotions_twitter, True, f'twitter_llm{args.prompt}')
all_data['twitter'] = res

emotion_H_tests_dialogue = get_test_rep(data_dialogue, emotions_dialogue, rep_readers)
res = get_acc(emotion_H_tests_dialogue, rep_readers, emotions_dialogue, True, f'dialogue_llm{args.prompt}')
all_data['dialogue'] = res

print(f">>> All LLM {args.prompt} cross plots saved!")

df = extract_max_scores(all_data)

# Save as CSV
df.to_csv(f"results/PCA_llm{args.prompt}_cross_layer_scores.csv", index=False)
torch.save(all_data,f'results/PCA_reading_vecs/results_all_data_llm{args.prompt}.pt')

print(df.head())
print(f">>> All LLM-{args.prompt} data saved!")

