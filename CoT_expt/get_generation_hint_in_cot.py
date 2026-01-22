import pandas as pd
import numpy as np
import json
import re
import pandas as pd
import json
import numpy as np
import random
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import time
random.seed(2025)
import os
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(
        description='prompting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-model_path',type=str, default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('-df_path',type=str, help='path to file with the cot generations')
parser.add_argument('-gpu_id',type=str, default='0')
args = parser.parse_args()

# To make only GPU device 0 visible
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

df= pd.read_json("data/EU.jsonl", orient='records', lines=True)
df = df[df['language']=='en']
df['ans_emotion_index'] = [l.index(i) for l,i in zip(list(df['emotion_choices']), list(df['emotion_label']))]
df['ans_choice_index'] = [l.index(i) for l,i in zip(list(df['cause_choices']), list(df['cause_label']))]



def remove_pred_index(cols,df):
    preds = {}
    pattern = re.compile(r"(?:selected choice index[:=]|chosen choice index|selected choice|=\s?)[^\d]*(\d.*)",re.IGNORECASE)

    for c in cols:
        tmp = [pattern.sub("", str(i)) for i in list(df[c])] # Basically erasing everything after, included the selected choice part
        preds[c] = tmp
    return preds


df2 = pd.read_json(args.df_path, lines=True)
# Col of interes - raw_hint_cot
df2_cols = [c for c in list(df2.columns) if 'raw' in c and 'cot' in c and 'hint' in c]
# Modifying the selected index to empty
cot_data = remove_pred_index(df2_cols, df2)



model = AutoModelForCausalLM.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=425, temperature=0, do_sample=False)
model_name = args.model_path.split('/')[-1]#'Llama-3.1-8B-I'

def parse_generation(col_name,predictions, df=df, model_name=model_name):
    tmp = [re.findall(r"\d",i) for i in predictions]
    df[f'pred_withouthint_{model_name}_{col_name}'] = [int(i[-1]) if len(i)>0 else -1 for i in tmp]

    return df



template = "Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. A reasoning trace is provided to help you choose the answer. Output only the appropriate choice number nothing else.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nReasoning:{reasoning}\nOutput:"



def get_generations(prompt_template, col_name, df=df):
    predictions = []
    # reference = []
    reasoning_data = cot_data[col_name]

    for rno,row in tqdm(df.iterrows(), total=len(df)):
        scenario = row['scenario']
        subject =  row['subject']
        choices =  row['emotion_choices']
        reasoning_current  = reasoning_data[rno]

        indexed_choices = [str(i)+" :"+c.lower() for i,c in enumerate(choices)]

        prompt = prompt_template.format(scenario=scenario, subject=subject, indexed_choices=indexed_choices, reasoning=reasoning_current)
        if(rno<1):
            print(rno, prompt)

        # pass to model
        messages = [{"role": "user", "content": prompt},]
        response = pipe(messages)
        #get response
        prediction = response[0]['generated_text'][-1]['content']
        # save along with reference answer
        predictions.append(prediction)

    # if(ptype == 'cot'):
    df[f'raw_{model_name}_{col_name}'] = predictions
    df = parse_generation(col_name,predictions, df=df)
    acc = np.mean([int(i)==int(j) for i,j in zip(list(df[f'pred_withouthint_{model_name}_{col_name}']), list(df['ans_emotion_index']))])
    print(f"Acc for ,{model_name}, {col_name} = {acc}")
    print("*********************************************")
    
    return acc, df


col_names = list(cot_data.keys())
for col_name_ in col_names:
    acc, df = get_generations(template, col_name_, df=df)
    df.to_json(f"EU_pred_cot-hint-verify_{model_name}.jsonl",orient='records', lines=True)
