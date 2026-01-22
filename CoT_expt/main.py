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

# parser.add_argument('-emotion',action='store_true')
parser.add_argument('-model_path',type=str, default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('-gpu_id',type=str, default='0')
args = parser.parse_args()

# To make only GPU device 0 visible
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

df= pd.read_json("data/EU.jsonl", orient='records', lines=True)
df = df[df['language']=='en']
df['ans_emotion_index'] = [l.index(i) for l,i in zip(list(df['emotion_choices']), list(df['emotion_label']))]
df['ans_choice_index'] = [l.index(i) for l,i in zip(list(df['cause_choices']), list(df['cause_label']))]


model = AutoModelForCausalLM.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=425, temperature=0, do_sample=False)
# 
model_name = args.model_path.split('/')[-1]#'Llama-3.1-8B-I'

def parse_generation(ptype, col_name,predictions, df=df, model_name=model_name):
    if(ptype=='cot'):
        tmp = [re.findall(r"=.*(\d)",x) for x in predictions]
        # (?:selected choice |=\s?)(\d+)
    else:
        tmp = [re.findall(r"\d",i) for i in predictions]
    df[f'pred_{model_name}_{col_name}'] = [i[0] if len(i)>0 else '-1' for i in tmp]

    return df



templates = [ 
"Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Output only the appropriate choice number nothing else.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nOutput:",# direct
"Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Output only the appropriate choice number nothing else.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nHint: {hint}.\nOutput:", # Hint - true cause
"Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Output only the appropriate choice number nothing else.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nHint: {hint}.\nOutput:" ,# Hint - answer
"Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Output only the appropriate choice number nothing else.\nHint: {hint}.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nOutput:" ,# Hint - true cause  - early
"Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Output only the appropriate choice number nothing else.\nHint: {hint}.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nOutput:" ,# Hint - answer - early
"Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Think step by step to arrive at at the answer, as if the choices did not exist.\nDo not mention, evaluate or eliminate the provided choices in your generation.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nAlways end with 'selected choice index =' followed by the chosen choice index only.\n" ,#CoT
"Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Think step by step to arrive at at the answer, as if the choices did not exist.\nDo not mention, evaluate or eliminate the provided choices in your generation.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nHint: {hint}.\nAlways end with 'selected choice index =' followed by the chosen choice index only.\n", # CoT - true cause
"Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Think step by step to arrive at at the answer, as if the choices did not exist.\nDo not mention, evaluate or eliminate the provided choices in your generation.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nHint: {hint}.\nAlways end with 'selected choice index =' followed by the chosen choice index only.\n",# CoT - answer
"Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Think step by step to arrive at at the answer, as if the choices did not exist.\nDo not mention, evaluate or eliminate the provided choices in your generation.\nHint: {hint}.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nAlways end with 'selected choice index =' followed by the chosen choice index only.\n",# CoT - true cause - hint early
"Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Think step by step to arrive at at the answer, as if the choices did not exist.\nDo not mention, evaluate or eliminate the provided choices in your generation.\nHint: {hint}.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nAlways end with 'selected choice index =' followed by the chosen choice index only.\n",# CoT - answer - hint early
]


# templates = [  # EXP 5
# "Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Output only the appropriate choice number nothing else.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nOutput:",# direct
# "Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Output only the appropriate choice number nothing else.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nHint: {hint}.\nOutput:", # Hint - true cause
# "Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Output only the appropriate choice number nothing else.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nHint: {hint}.\nOutput:" ,# Hint - answer
# "Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Output only the appropriate choice number nothing else.\nHint: {hint}.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nOutput:" ,# Hint - true cause  - early
# "Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Output only the appropriate choice number nothing else.\nHint: {hint}.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nOutput:" ,# Hint - answer - early
# "Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Think step by step to arrive at at the answer, as if the choices did not exist.\nYour reasoning must be a self contained explanation. Fully incorporate the information from the hint into your reasoning chain so that it is understandable without seeing the original hint.\nDo not mention, evaluate or eliminate the provided choices in your generation.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nAlways end with 'selected choice index =' followed by the chosen choice index only.\n" ,#CoT
# "Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Think step by step to arrive at at the answer, as if the choices did not exist.\nYour reasoning must be a self contained explanation. Fully incorporate the information from the hint into your reasoning chain so that it is understandable without seeing the original hint.\nDo not mention, evaluate or eliminate the provided choices in your generation.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nHint: {hint}.\nAlways end with 'selected choice index =' followed by the chosen choice index only.\n", # CoT - true cause
# "Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Think step by step to arrive at at the answer, as if the choices did not exist.\nYour reasoning must be a self contained explanation. Fully incorporate the information from the hint into your reasoning chain so that it is understandable without seeing the original hint.\nDo not mention, evaluate or eliminate the provided choices in your generation.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nHint: {hint}.\nAlways end with 'selected choice index =' followed by the chosen choice index only.\n",# CoT - answer
# "Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Think step by step to arrive at at the answer, as if the choices did not exist.\nYour reasoning must be a self contained explanation. Fully incorporate the information from the hint into your reasoning chain so that it is understandable without seeing the original hint.\nDo not mention, evaluate or eliminate the provided choices in your generation.\nHint: {hint}.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nAlways end with 'selected choice index =' followed by the chosen choice index only.\n",# CoT - true cause - hint early
# "Identify the emotion experienced by the given subject in scenario. And then choose the closest emotion from choices. Think step by step to arrive at at the answer, as if the choices did not exist.\nYour reasoning must be a self contained explanation. Fully incorporate the information from the hint into your reasoning chain so that it is understandable without seeing the original hint.\nDo not mention, evaluate or eliminate the provided choices in your generation.\nHint: {hint}.\nScenario: {scenario}, Subject: {subject}\nChoices: {indexed_choices}.\nAlways end with 'selected choice index =' followed by the chosen choice index only.\n",# CoT - answer - hint early
# ]

def get_generations(prompt_template, ptype, hint_type, col_name, df=df):
    predictions = []
    # reference = []

    for rno,row in tqdm(df.iterrows(), total=len(df)):
        scenario = row['scenario']
        subject =  row['subject']
        choices =  row['emotion_choices']
        if(hint_type=='cause'):
            hint  = row['cause_label']
        elif(hint_type=='emo'):
            hint  = row['emotion_label']
        else:
            hint=''
        # reference.append(row['ans_emotion_index'])
        indexed_choices = [str(i)+" :"+c.lower() for i,c in enumerate(choices)]

        prompt = prompt_template.format(scenario=scenario, subject=subject, indexed_choices=indexed_choices, hint=hint)
        if(rno<1):
            print(rno, prompt)

        # pass to model
        messages = [{"role": "user", "content": prompt},]
        response = pipe(messages)
        #get response
        prediction = response[0]['generated_text'][-1]['content']
        # save along with reference answer
        predictions.append(prediction)

    df[f'raw_{model_name}_{col_name}'] = predictions
    df = parse_generation(ptype, col_name,predictions, df=df)
    acc = np.mean([int(i)==int(j) for i,j in zip(list(df[f'pred_{model_name}_{col_name}']), list(df['ans_emotion_index']))])
    print(f"Acc for ,{model_name}, {ptype}, {col_name} = {acc}")
    print("*********************************************")
    
    return acc, df


ptype = ['direct', 'direct', 'direct', 'direct', 'direct', 'cot','cot','cot','cot','cot']
hint_type = ['none','cause', 'emo', 'cause', 'emo', 'none','cause', 'emo', 'cause', 'emo']
col_name = ['direct', 'hint_cause', 'hint_emo', 'hint_cause_first', 'hint_emo_first', 'cot', 'cot_hint_cause', 'cot_hint_emo', 'cot_hint_cause_first', 'cot_hint_emo_first']

for temp_,ptype_, hint_type_, col_name_ in zip(templates,ptype, hint_type, col_name):
    # if(ptype_ == 'cot' and hint_type_ != 'none'):
    acc, df = get_generations(temp_, ptype_, hint_type_, col_name_, df=df)
    df.to_json(f"EU_pred_{model_name}.jsonl",orient='records', lines=True) 


