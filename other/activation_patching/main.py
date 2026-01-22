import pandas as pd
import os
import torch
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from LLMs.my_llama import LlamaForCausalLM
from LLMs.my_gemma2 import Gemma2ForCausalLM
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from prompt_manager import build_prompt
from utils import (Log, log_system_info, hf_login,TextDataset, emotion_to_token_ids, extract_hidden_states,
                   activation_patching)


##############################################################################################
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model_index", type=int, default=0, help="Index of the model to use, 0: Llama3.2_1B, 1: Llama3.1_8B, 2: Gemma-2-2b-it, 3: Gemma-2-9b-it, 4: Phi-3.5-mini-instruct, 5: Phi-3-medium-128k-instruct, 6: OLMo-1B-hf, 7: OLMo-7B-0724-Instruct-hf, 8: Ministral-8B-Instruct-2410, 9: Mistral-Nemo-Instruct-2407, 10: OLMo-2-1124-7B-Instruct, 11: OLMo-2-1124-13B-Instruct")
parser.add_argument("--bs", type=int, default=2, help="Batch Size")

parser.add_argument("--prompt_type", type=str, default='joy_sadness_0')
parser.add_argument("--probe_cls", type=str, default='default')

parser.add_argument("--given_path", type=str, default='default')

parser.add_argument("--prompt_version", type=str, default='P2')
parser.add_argument("--task_type", type=str)


parser.add_argument("--extract_hidden_states", action="store_true", default=False, help="Extract hidden states")
parser.add_argument("--emotion_probing", action="store_true", default=False, help="Perform emotion probing")
parser.add_argument("--activation_patching", action="store_true", default=False, help="Perform Activation Patching")

args = parser.parse_args()
##############################################################################################


log = Log(log_name='intervention')
logger = log.logger
log_system_info(logger)
hf_login(logger)
BATCH_SIZE = args.bs
HOOKED = True
device_map = 'cuda'

os.makedirs('outputs_baseline_expt6/', exist_ok=True)
os.makedirs(args.given_path, exist_ok=True)

# Handelling the 4 dataset  variations
#1. Twitter - emotion
#2. Crowd-eVent
# 3. Crowd-eVent-hidden
# 4. LLM gen data
# 5.Dialogue dataset

model_names =       ['meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct','google/Llama-3.1-8B-Instruct', 
                     'google/gemma-2-2b-it', 'google/gemma-2-9b-it' ]
model_short_names = ['Llama3.2_1B', 'Llama3.2_3B','Llama3.1_8B',
                     'gemma-2-2b-it', 'gemma-2-9b-it']

model_classes =     [LlamaForCausalLM, LlamaForCausalLM,LlamaForCausalLM,
                     Gemma2ForCausalLM, Gemma2ForCausalLM,]

model_name, model_short_name, model_class = list(zip(model_names, model_short_names, model_classes))[args.model_index]

save_prefix = ''
if not HOOKED:
    logger.info("Using an unhooked model ...")
    model_class = AutoModelForCausalLM
    save_prefix = 'UNHOOKED_'

logger.info(f"Model Name: {model_name}")

if args.prompt_type == 'joy_sadness_0': # default mode, no need to change path_prefix
    args.path_prefix = ''
else:
    args.path_prefix = args.prompt_type + '_'

if args.task_type == 'FirstWord':
    args.path_suffix = '_FirstWord'
else:
    args.path_suffix = ''

model_short_name =  args.path_prefix + model_short_name + args.path_suffix

os.makedirs(f'outputs_baseline_expt/{model_short_name}/{args.task_type}', exist_ok=True)

def get_data(data_type = args.task_type):
    if(data_type=='twitter'):
        train_data = pd.read_csv('datasets/emotion_train.csv')
        test_data = pd.read_csv('datasets/emotion_test.csv')
        inp_col = 'text'
        label_col = 'label'

    elif(data_type=='crowd'):
        train_data = pd.read_csv('datasets/crowd-event_train.csv')
        test_data = pd.read_csv('datasets/crowd-event_test.csv')
        emotions_list = ['anger','disgust', 'fear', 'joy', 'sadness', 'surprise']
        emotion_to_id = {emotion: i for i, emotion in enumerate(emotions_list)}
        id_to_emotion = {v: k for k, v in emotion_to_id.items()}
        train_data['emotion_id'] = train_data['emotion'].map(emotion_to_id).astype(int)
        test_data['emotion_id'] = test_data['emotion'].map(emotion_to_id).astype(int)
        inp_col = 'text'
        label_col = 'emotion_id'

    elif(data_type=='crowd-h'):
        train_data = pd.read_csv('datasets/crowd-event_train.csv')
        test_data = pd.read_csv('datasets/crowd-event_test.csv')
    
        emotions_list = ['anger','disgust', 'fear', 'joy', 'sadness', 'surprise']
        emotion_to_id = {emotion: i for i, emotion in enumerate(emotions_list)}
        id_to_emotion = {v: k for k, v in emotion_to_id.items()}
        train_data['emotion_id'] = train_data['emotion'].map(emotion_to_id).astype(int)
        test_data['emotion_id'] = test_data['emotion'].map(emotion_to_id).astype(int)
        
        inp_col = 'hidden_emo_text'
        label_col = 'emotion_id'

    elif(data_type=='llm'):
        train_data = pd.read_csv('datasets/rep_llm_data/train.csv')
        test_data = pd.read_csv('datasets/rep_llm_data/test.csv')
        emotions_list = ['anger','disgust','fear','happiness','sadness','surprise']
        emotion_to_id = {emotion: i for i, emotion in enumerate(emotions_list)}
        
        id_to_emotion = {v: k for k, v in emotion_to_id.items()}
        train_data['emotion_id'] = train_data['label'].map(emotion_to_id).astype(int)
        test_data['emotion_id'] = test_data['label'].map(emotion_to_id).astype(int)

        inp_col = 'text'
        label_col = 'emotion_id'

    elif(data_type=='dialogue'):
        train_data = pd.read_csv('datasets/dailydialog_test_processed_filtered.json')
        test_data = pd.read_csv('datasets/dailydialog_test_processed_filtered.json')
        emotions_list = ['anger','disgust', 'fear', 'joy', 'sadness', 'surprise']
        emotion_to_id = {emotion: i for i, emotion in enumerate(emotions_list)}
        id_to_emotion = {v: k for k, v in emotion_to_id.items()}

        train_data['emotion'] = train_data['emotion'].replace('no-emotion', 'neutral')
        test_data['emotion'] = test_data['emotion'].replace('no-emotion', 'neutral')
        
        train_data['emotion_id'] = train_data['emotion'].map(emotion_to_id).astype(int)
        test_data['emotion_id'] = test_data['emotion'].map(emotion_to_id).astype(int)


        inp_col = 'conversation'
        label_col = 'emotion_id'


    if '_' in args.prompt_type:
        shots = args.prompt_type.split('_')[:-1]
        prompt_index = int(args.prompt_type.split('_')[-1])
    else:
        shots = []
        prompt_index = int(args.prompt_type)

    func = build_prompt(shots=shots, prompt_index=prompt_index, version=args.prompt_version)
    train_data['input_text'] = train_data[inp_col].apply(func)
    test_data['input_text'] = test_data[inp_col].apply(func)
    print(list(test_data['input_text'])[0])

    train_data['emotion'] = list(train_data[label_col])

    labels_train = torch.from_numpy(train_data[[label_col]].to_numpy())
    labels_test = torch.from_numpy(test_data[[label_col]].to_numpy())

    dataset_train = TextDataset(train_data['input_text'].tolist(), labels_train)
    dataloader_train = DataLoader(dataset_train, batch_size = BATCH_SIZE, shuffle=False)

    dataset_test = TextDataset(test_data['input_text'].tolist(), labels_test)
    dataloader_test = DataLoader(dataset_test, batch_size = BATCH_SIZE, shuffle=False)

    return dataloader_train, dataloader_test, labels_train, labels_test, train_data

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
if tokenizer.pad_token is None:
    logger.info("Adding padding token to the tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

model = model_class.from_pretrained(model_name, device_map=device_map)

num_params = sum(p.numel() for p in model.parameters())
size_on_memory = sum(p.numel() * p.element_size() for p in model.parameters())
logger.info(f"Loaded model '{model_name}' with {num_params}  parameters ({size_on_memory / (1024 ** 2):.2f} MB)")
logger.info(f"Model configuration: {model.config}")

dataloader_train, dataloader_test, labels_train, labels_test, train_data = get_data(data_type = args.task_type)
emotions_list = ["happiness", "sadness", "anger", "fear", "disgust", "surprise", "joy", "love"] #list(train_data['emotion'].unique())
emotions_to_tokenized_ids = emotion_to_token_ids(emotions_list, tokenizer)




if args.extract_hidden_states:
    logger.info("------------------------------ Extracting Hidden States ------------------------------")
    extraction_locs = [3, 6, 7]
    # extraction_locs = [7]
    extraction_tokens = [-1]#, -2, -3, -4, -5]
    extraction_layers = list(range(model.config.num_hidden_layers))

    all_hidden_states = extract_hidden_states(dataloader_train, tokenizer, model, logger, extraction_locs=extraction_locs, extraction_layers=extraction_layers, extraction_tokens = extraction_tokens)
    all_hidden_states_test = extract_hidden_states(dataloader_test, tokenizer, model, logger, extraction_locs=extraction_locs, extraction_layers=extraction_layers, extraction_tokens = extraction_tokens)
    # torch.save(all_hidden_states, f'outputs/{model_short_name}/hidden_states_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt')


# if args.emotion_probing:
#     logger.info("--------------------------------- Emotion Probing ---------------------------------")

#     extraction_locs = [3, 6, 7]
#     # extraction_locs = [7]
#     extraction_tokens = [-1]#, -2, -3, -4, -5]
#     extraction_layers = list(range(model.config.num_hidden_layers))

#     if not args.extract_hidden_states:
#         try:
#             all_hidden_states = torch.load(f'outputs_baseline_expt6/{model_short_name}/{args.task_type}/hidden_states_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt', weights_only=False)
#         except:
#             raise Exception("Hidden states not found, run the code again with --extract_hidden_states")
        
    
#     size_on_memory = all_hidden_states.element_size() * all_hidden_states.numel()
#     logger.info(f"Hidden states tensor size: {size_on_memory / (1024 ** 2):.2f} MB")

#     results = {}
#     for i, layer in tqdm(enumerate (extraction_layers), total=len(extraction_layers)):
#         results[layer] = {}
#         for j, loc in enumerate (extraction_locs):
#             results[layer][loc] = {}
#             for k, token in enumerate (extraction_tokens):
#                 results[layer][loc][token] = probe_classification_local(all_hidden_states[:, i, j, k], labels_train[:, 0],all_hidden_states_test[:, i, j, k], labels_test[:, 0], return_weights=False)
                

#     # print(f"Results train-test split {args.prompt_version}: ", results)

#     new_res = {}
#     for layer in results.keys():
#         new_res[layer] = {}
#         for loc in extraction_locs:
#             new_res[layer][loc] = [results[layer][loc][-1]['accuracy_test'].item(),results[layer][loc][-1]['accuracy_train'].item()]
#     # print(layer,res[layer][7][-1]['accuracy_train'])
#     print(new_res)
#     new_res['model'] = model_short_name
#     new_res['prompt'] = args.prompt_version
#     new_res['dataset'] = args.task_type

#     torch.save(results, f'outputs_baseline_expt6/{model_short_name}/{args.task_type}/emotion_probing_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}_{args.prompt_version}.pt')

#     with open('outputs_baseline_expt6/plot_res.json','a') as f:
#         json.dump(new_res, f, indent=2)


if args.activation_patching:
    logger.info("------------------------------ Performing Activation Patching ------------------------------")
    num_experiments = 200
    span = 5
    layers_ = list(range(span // 2, model.config.num_hidden_layers - span // 2))

    for locs in [[3], [6], [7]]:
        for intervention_tokens in [[-1]]: #, [-2], [-3], [-4], [-5]
            patching_results = {}
            
            if(os.path.exists(f'outputs_baseline_expt/patching/{model_short_name}/{args.task_type}/patching_results_layers_{layers_}_locs_{locs}_span_{span}_token_{intervention_tokens}_{args.prompt_version}.pt')):
                print(f'skipping /{model_short_name}/{args.task_type}/patching_results_layers_{layers_}_locs_{locs}_span_{span}_token_{intervention_tokens}_{args.prompt_version}.pt')
                continue

            for center_layer in tqdm(layers_, total=len(layers_)):
                layers = list(range(center_layer - span // 2, center_layer + span // 2 + 1))

                patching_results[center_layer] = {}
                for i in range(num_experiments):
                    # sample a random source sentence and target sentence with different emotions
                    source_sentence = train_data.sample(1)
                    target_sentence = train_data.sample(1)
                    while source_sentence['emotion'].values[0] == target_sentence['emotion'].values[0]:
                        target_sentence = train_data.sample(1)

                    source_sentence, source_emotion = source_sentence['input_text'].values[0], source_sentence['emotion'].values[0]
                    target_sentence, target_emotion = target_sentence['input_text'].values[0], target_sentence['emotion'].values[0]

                    patching_results[center_layer][i] = activation_patching(source_sentence, target_sentence, tokenizer, model, logger, intervention_layers=layers, intervention_locs=locs, ids_to_pick=emotions_to_tokenized_ids, intervention_tokens=intervention_tokens)


            # outputs/{model_short_name}/{args.task_type}/
            os.makedirs(f'outputs_baseline_expt/patching/{model_short_name}/{args.task_type}', exist_ok=True)

            torch.save(patching_results, f'outputs_baseline_expt/patching/{model_short_name}/{args.task_type}/patching_results_layers_{layers_}_locs_{locs}_span_{span}_token_{intervention_tokens}_{args.prompt_version}.pt')





