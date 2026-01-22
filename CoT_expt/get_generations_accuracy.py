# python get_generations_accuracy.py -df_path "EU_pred_gemma-2-9b-it.jsonl"
import numpy as np
import pandas as pd
import re
from typing import List, Union, Tuple


import argparse

parser = argparse.ArgumentParser(
        description='prompting_parse_generations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument('-emotion',action='store_true')
parser.add_argument('-df_path',type=str) 
args = parser.parse_args()

df = pd.read_json(args.df_path, lines=True)
col = list(df.columns)
gen_cols = [re.findall(r"raw.*",c)[0] for c in col if len(re.findall(r"raw.*",c))]
cot_cols = [c for c in gen_cols if 'cot' in c]
cot_cols.sort()
direct_cols = list(set(gen_cols) - set(cot_cols))
direct_cols.sort()

ground = list(df['ans_emotion_index'])
choices = list(df['emotion_choices'])

def _perform_search(
    sentences_to_search: List[str],
    parsed_options: List[dict]
) -> Tuple[List[str], List[int]]:
    """Helper function to run the matching logic on a given list of sentences."""
    full_matches = []
    partial_matches = {}

    for sentence in reversed(sentences_to_search):
        sentence_lower = sentence.lower()
        sentence_words_list = re.findall(r'\b\w+\b', sentence_lower)
        sentence_words_set = set(sentence_words_list)
        # sentence_words = {re.sub(r'[^\w\s-]', '', word) for word in sentence_lower.split()}
        
        for option in parsed_options:
            keywords = option['keywords']
            found_keywords = keywords.intersection(sentence_words_set)

            if len(found_keywords) == len(keywords): # Full match
                sentence_word_list = [re.sub(r'[^\w\s-]', '', word) for word in sentence_lower.split()]
                indices = [i for i, word in enumerate(sentence_word_list) if word in keywords]
                proximity_score = max(indices) - min(indices) if indices and max(indices) > min(indices) else 0
                full_matches.append({'option': option, 'score': proximity_score})
            elif len(found_keywords) > 0: # Partial match
                partial_matches[option['original']] = option['index']
    
    if full_matches:
        best_full_match = min(full_matches, key=lambda x: x['score'])
        best_option = best_full_match['option']
        return ([best_option['original']], [best_option['index']], "full_match")
    
    if partial_matches:
        sorted_strings = sorted(list(partial_matches.keys()))
        sorted_indices = [partial_matches[s] for s in sorted_strings]
        return (sorted_strings, sorted_indices, "partial_match")
        
    return ([-1], [-1], "no_match")


def find_best_or_partial_match(
    options_list: List[str], 
    long_text: str,
    search_window: int = 3
) -> Tuple[List[str], List[int]]:
    """
    Finds matches using a tiered search logic.
    """
    parsed_options = []
    for i, option_str in enumerate(options_list):
        keywords = {word.strip().lower() for word in option_str.split(' & ')}
        parsed_options.append({'original': option_str, 'keywords': keywords, 'index': i})

    sentences = re.split(r'(?<=[.?!])\s+', long_text.strip())

    # --- TIER 1: Search in the last `search_window` sentences ---
    strings, indices, match_case = _perform_search(sentences[-search_window:], parsed_options)
    if strings:
        return (strings, indices, match_case)

    return ([-1], [-1], "no_match")

def pred_index(cot_cols):
    preds = {}
    pattern = re.compile(r"(?:selected choice index[:=]|chosen choice index|selected choice|=\s?)[^\d]*(\d+)", re.IGNORECASE)

    for c in cot_cols:
        preds[c] = []
        col_data = list(df[c])
        for text in col_data:
            match = pattern.search(text)
            if match:
                preds[c].append(match.group(1))
            else:
                preds[c].append(-1)
    return preds

direct_preds_dict = pred_index(cot_cols) # All col key, and value = pred index list
options_list = list(df['emotion_choices'])
df_len = len(df)

print("ALL DIRECT GENERATION RESULTS")
for c in direct_cols:
    text_gen = list(df[c])
    tmp = [re.findall(r"\d",str(i)) for i in text_gen]
    extracted_predictions = [int(i[-1]) if len(i)>0 else '-1' for i in tmp]
    acc = np.mean([int(i)==int(j) for i,j in zip(extracted_predictions, ground)])
    missed = [1 if i==-1 else 0 for i in extracted_predictions]
    print(f"{c} | Missed = {round(np.sum(missed)/df_len, 4)}, Accuracy = {round(acc,4)}")


text_preds_dict = {}
for c in cot_cols:
    text_preds_dict[c] = {}
    col_data = list(df[c])
    tmp_text, tmp_index, tmp_case = [],[],[]
    for opt, text in zip(options_list, col_data):
        pred_text, pred_index, pred_case = find_best_or_partial_match(opt, text)
        tmp_text.append(pred_text)
        tmp_index.append(pred_index)
        tmp_case.append(pred_case)
    text_preds_dict[c]['pred_text'] = tmp_text
    text_preds_dict[c]['pred_index'] = tmp_index
    text_preds_dict[c]['matched_case'] = tmp_case

print("ALL COT GENERATION RESULTS")

# direct predicted options case
print("Direct index predictions:")
for col in direct_preds_dict.keys():
    pred = direct_preds_dict[col]
    acc = 0
    missed = 0
    for g,p in zip(ground, pred):
        if(int(g) == int(p)):
            acc +=1
        elif(int(p)==-1):
            missed +=1
    print(f"{col} | Missed = {round(missed/df_len, 4)}, Accuracy = {round(acc/df_len,4)}")

print("Option text match predictions")
for col in text_preds_dict.keys():
    pred = text_preds_dict[col]['pred_index'] # list
    pred_case = text_preds_dict[col]['matched_case'] # string
    acc_full = 0
    acc_partial_full = 0
    missed = 0
    for g,p, pc in zip(ground, pred, pred_case):
        if(int(g) in p and pc == 'partial_match'):
            acc_partial_full +=1
            if(pc == 'full_match'):
                acc_full += 1
        elif(-1 in p):
            missed +=1
    print(f"{col} | Missed = {round(missed/df_len, 4)}, Full Accuracy = {round(acc_full/df_len,4)}, Partial Accuracy = {round(acc_partial_full/df_len,4)}")

print("Starting with direct index predictions and adding the full/partial match as needed to reduce missed")
for col in direct_preds_dict.keys():
    pred = direct_preds_dict[col]
    pred2 = text_preds_dict[col]['pred_index']
    pred2_case = text_preds_dict[col]['matched_case']
    acc = 0
    acc2 = 0
    missed = 0
    for g, p1, p2, p2_case in zip(ground, pred, pred2, pred2_case):
        if(int(g) == int(p1)):
            acc +=1
        elif(int(g) in p2 and p2_case=='full_match'):
            acc +=1
        elif(int(g) in p2 and p2_case=='partial_match'):
            acc2 +=1
        elif (-1 in p2 and -1==int(p1)):
            missed +=1
    print(f"{col} | Missed = {round(missed/df_len, 4)}, index+full Accuracy = {round((acc)/df_len,4)}, index+full+partial Accuracy = {round((acc+acc2)/df_len,4)}")

print("Starting with full predictions and adding direct and partial match as needed to reduce missed")
for col in direct_preds_dict.keys():
    pred = direct_preds_dict[col]
    pred2 = text_preds_dict[col]['pred_index']
    pred2_case = text_preds_dict[col]['matched_case']
    acc = 0
    acc2 = 0
    missed = 0
    for g, p1, p2, p2_case in zip(ground, pred, pred2, pred2_case):
        
        if(int(g) in p2 and pred2_case == 'full_match'):
            acc +=1
        elif(int(g) == int(p1)):
            acc +=1
        elif(int(g) in p2 and p2_case =='partial_match'):
            acc2 +=1
        elif (-1 in p2 and -1==int(p1)):
            missed +=1
    print(f"{col} | Missed = {round(missed/df_len, 4)}, Full_text+index Accuracy = {round(acc/df_len,4)}, Full_text+index+partial Accuracy = {round((acc+acc2)/df_len,4)}")

# for cases where both full/partial text present and pred index present, checking if model got emotion -> option right
print("For cases where both full/partial text present and pred index present, checking if model got emotion -> option right")
for col in direct_preds_dict.keys():
    pred = direct_preds_dict[col]
    pred2 = text_preds_dict[col]['pred_index']
    index_text_match = 0
    index_text_missmatch = 0
    missed = 0
    for g, p1, p2 in zip(ground, pred, pred2):
        if(-1 not in p2 and int(p1) in p2):
            index_text_match +=1
        elif(-1 not in p2 and int(p1) not in p2):
            index_text_missmatch +=1
        elif (-1 in p2 and -1==int(p1)):
            missed +=1
    tmp_total = index_text_match + index_text_missmatch
    print(f"{col} | Missed = {round(missed/df_len, 4)}, For cases with both emotion text and choice index {tmp_total} cases, index_text_match = {round(index_text_match/tmp_total,4)}")

# Finding edge missed cases to improve regex above:
print("Edge missed cases to potentially improve regex...")
for col in direct_preds_dict.keys():
    pred = direct_preds_dict[col]
    pred2 = text_preds_dict[col]['pred_index']
    cot_text = list(df[col])#choices
    print(f"-----------------------{col}-------------------------")
    for i,j,k,l in zip(pred, pred2, cot_text, choices):
        if(int(i)==-1 and -1 in j):
            print(k[-300:])
            print(l)


def get_index_pred(col_data, col_name):
    if 'cot' in col_name:
        extracted_predictions = []
        pattern = re.compile(r"(?:selected choice index[:=]|chosen choice index|selected choice|=\s?)[^\d]*(\d+)", re.IGNORECASE)
        for text in col_data:
            match = pattern.search(text)
            if match:
                extracted_predictions.append(int(match.group(1)))
            else:
                extracted_predictions.append(-1)

    else:
        tmp = [re.findall(r"\d",str(i)) for i in col_data]
        extracted_predictions = [int(i[-1]) if len(i)>0 else '-1' for i in tmp]
    return extracted_predictions

# print(Comparing case when the direct and cot pred are same) -> to test if the CoT actually helped change?
print("Comparing if accuracy improved with CoT? ... INDEX MATCH!")
for cc, dc in zip(cot_cols,direct_cols):
    # Taking the index predicted
    cc_pred_index = get_index_pred(list(df[cc]), cc)
    dc_pred_index = get_index_pred(list(df[dc]), dc)

    # when  not equal, which is correct - direct, cot, or both wrong
    same_pred = 0
    direct_correct_cot_wrong = 0
    direct_wrong_cot_correct = 0
    both_diff_wrong = 0
    both_same_wrong = 0
    both_same_correct = 0
    any_missed = 0
    for i,j,g in zip(dc_pred_index,cc_pred_index,  ground):
            if(i==j):
                same_pred +=1
                if(i!=g):
                    both_same_wrong +=1
                else:
                    both_same_correct +=1
            else:
                # if(i!=-1 and j!=-1):
                if(i==g):
                        direct_correct_cot_wrong += 1
                elif(j==g):
                        direct_wrong_cot_correct +=1
                else:
                        both_diff_wrong +=1
            if(i==-1 or j==-1):
                    any_missed +=1
    print(f"{cc}, {dc}, Same predictions = {round(same_pred/len(df),3)}, correct = {round(both_same_correct/len(df),3)}, wrong = {round(both_same_wrong/len(df),3)}")
    print(f"{cc}, {dc}, Diff valid predictions, direct_correct_cot_wrong = {round(direct_correct_cot_wrong/len(df),3)}, direct_wrong_cot_correct = {round(direct_wrong_cot_correct/len(df),3)}, both diff wrong = {round(both_diff_wrong/len(df),3)}")
    print(f" Any missed in this missmatch: {round(any_missed/len(df),3)}")

print("=========================================================================================")
print("Comparing if accuracy improved with CoT?  - CoT-lenient... INDEX + FULL OPTION MATCH!")


for cc, dc in zip(cot_cols,direct_cols):
    # Taking the index predicted
    cc_pred_index_direct = get_index_pred(list(df[cc]), cc)
    pred = direct_preds_dict[cc]
    tmp = [i==j for i,j in zip(cc_pred_index_direct, pred)]
    print(f"cc_pred == pred? {np.mean(tmp)}")
    cc_pred_index = text_preds_dict[col]['pred_index']
    cc_pred_case = text_preds_dict[col]['matched_case']

    dc_pred_index = get_index_pred(list(df[dc]), dc)
    newc = []
    cot_index_wrong_text_correct = 0

    for i,j,k,g in zip(cc_pred_index_direct, cc_pred_index, cc_pred_case, ground):
        if(int(i) ==int(g)):
            newc.append(i)
        elif(int(i)==-1 and int(j[-1])!=-1 and k=='full_match'):
            newc.append(j[-1])
        if(int(i)!=-1 and int(i)!=int(g) and int(j[-1])==int(g) and k=='full_match'):
            cot_index_wrong_text_correct +=1
    print(f"{cc}, cot_index_wrong_text_correct: {round(cot_index_wrong_text_correct /len(df), 3)}")

    same_pred = 0
    direct_correct_cot_wrong = 0
    direct_wrong_cot_correct = 0
    both_diff_wrong = 0
    both_same_wrong = 0
    both_same_correct = 0
    any_missed = 0
    for i,j,g in zip(dc_pred_index,newc,  ground):
            if(i==j):
                same_pred +=1
                if(i!=g):
                    both_same_wrong +=1
                else:
                    both_same_correct +=1
            else:
                if(i==g):
                        direct_correct_cot_wrong += 1
                elif(j==g):
                        direct_wrong_cot_correct +=1
                else:
                        both_diff_wrong +=1
                if(i==-1 or j==-1):
                    any_missed +=1
    print(f"{cc}, {dc}, Same predictions = {round(same_pred/len(df),3)}, correct = {round(both_same_correct/len(df),3)}, wrong = {round(both_same_wrong/len(df),3)}")
    print(f"{cc}, {dc}, Diff valid predictions, direct_correct_cot_wrong = {round(direct_correct_cot_wrong/len(df),3)}, direct_wrong_cot_correct = {round(direct_wrong_cot_correct/len(df),3)}, both diff wrong = {round(both_diff_wrong/len(df),3)}")
    print(f" Any missed in this missmatch: {round(any_missed/len(df),3)}")

