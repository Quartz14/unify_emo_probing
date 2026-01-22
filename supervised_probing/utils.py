import os
import subprocess
from datetime import datetime

from functools import partial
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import random

import cupy as cp
from cuml.svm import LinearSVC
from cuml.linear_model import LogisticRegression


# Define the dataset class for handling text data
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        # self.labels = labels
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        return text, label


# Log class to handle logging activities
class Log:
    def __init__(self, log_name='probe'):
        filename = f'{log_name}_date-{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}.txt'
        os.makedirs('logs', exist_ok=True)
        self.log_path = os.path.join('logs/', filename)
        self.logger = self._setup_logging()

    def _setup_logging(self):
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler(self.log_path),
                                logging.StreamHandler()
                            ])
        return logging.getLogger()


def log_system_info(logger):
    """
    Logs system memory and GPU details.
    """

    def run_command(command):
        """
        Runs a shell command and returns its output.

        Args:
        - command (list): Command and arguments to execute.

        Returns:
        - str: Output of the command.
        """
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return result.stderr

    gpu_info = run_command(['nvidia-smi'])

    if os.name == 'nt':  # windows system
        pass
    else:
        memory_info = run_command(['free', '-h'])
        logger.info("Memory Info:\n" + memory_info)

    logger.info("GPU Info:\n" + gpu_info)


def hf_login(logger):
    load_dotenv()
    try:
        # Retrieve the token from an environment variable
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token is None:
            logger.error("Hugging Face token not set in environment variables.")
            return

        # Attempt to log in with the Hugging Face token
        login(token=token)
        logger.info("Logged in successfully to Hugging Face Hub.")
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")



def probe_classification_local(given_path, layer, train_hidden_states, train_labels, test_hidden_states, test_labels,return_weights=False, Normalize_X = False, reg_strength = 1.0, fit_intercept = True, probe_type='default'):
    if len(train_labels.shape) == 2:
        train_labels = train_labels[:, 0]
    if len(test_labels.shape) == 2:
        test_labels = test_labels[:, 0]

    X_train = train_hidden_states.reshape(train_hidden_states.shape[0], -1)
    X_test = test_hidden_states.reshape(test_hidden_states.shape[0], -1)

    # Y_train = train_labels
    Y_test = test_labels

    Y_train = cp.asarray(train_labels, dtype=cp.int32)
    X_train = cp.asarray(X_train, dtype=cp.float32)
    X_test  = cp.asarray(X_test, dtype=cp.float32)

    if(probe_type=='default'):
        # net = LogisticRegression(C = 1 / reg_strength, fit_intercept=fit_intercept)
        net = LogisticRegression(
                C=1,
                max_iter=1000,
                fit_intercept=True,
                tol=1e-4,
                penalty='l2',       # cuML supports 'none' and 'l2'
                solver='qn',        # quasi-Newton solver is default for cuML LR
                )
        # net.fit(X_train, Y_train)
    else:
        net = LinearSVC(C = 1 / reg_strength, fit_intercept=True, max_iter=1000, penalty='l2', loss='squared_hinge')
    
    net.fit(X_train, Y_train)

    y_pred_train = net.predict(X_train)
    y_pred_test = net.predict(X_test)

    if isinstance(Y_train, np.ndarray):
        Y_train = torch.tensor(Y_train)
        Y_test = torch.tensor(Y_test)
    
    if isinstance(y_pred_train, np.ndarray):
        y_pred_train = torch.tensor(y_pred_train)
        y_pred_test = torch.tensor(y_pred_test)

    accuracy_train = torch.tensor(Y_train.get() == y_pred_train.get()).float().mean()

    # accuracy_test = (Y_test == y_pred_test).float().mean()
    accuracy_test = torch.tensor(Y_test == y_pred_test.get()).float()
    #torch.save(y_pred_test.get(), given_path+str(layer)+".pt")


    res = {'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test.mean()}
    # print(res)
    if return_weights:
        res['weights'] = net.coef_
        res['bias'] = net.intercept_

    return res



extraction_locations = {1: "model.layers.[LID].hook_initial_hs",
                        2: "model.layers.[LID].hook_after_attn_normalization",
                        3: "model.layers.[LID].hook_after_attn",
                        4: "model.layers.[LID].hook_after_attn_hs",
                        5: "model.layers.[LID].hook_after_mlp_normalization",
                        6: "model.layers.[LID].hook_after_mlp",
                        7: "model.layers.[LID].hook_after_mlp_hs",
                        8: "model.layers.[LID].self_attn.hook_attn_heads",
                        9: "model.final_hook",
                        10: "model.layers.[LID].self_attn.hook_attn_weights",
                        }

def name_to_loc_and_layer(name):
    layer = int(name.split("model.layers.")[1].split(".")[0])
    loc_suffixes = {v.split('.')[-1]:k for k,v in extraction_locations.items()}
    loc = loc_suffixes[name.split(".")[-1]]
    
    return loc, layer

def extract_from_cache(cache_dict_, extraction_layers=[0, 1],
                          extraction_locs=[1, 7],
                          extraction_tokens=[-1],concat=False):
    return_value = []

    for layer in extraction_layers:
        return_value.append([])
        for el_ in extraction_locs:
            el = extraction_locations[el_].replace("[LID]", str(layer))
            if el_ != 10: # attention weights should be treated differently
                if(concat):
                    tmp = cache_dict_[el]# (batch,inp_len)
                    x_mean = tmp.mean(dim=1,keepdim=True)
                    x_max = tmp.max(dim=1,keepdim=True).values
                    x_min = tmp.min(dim=1,keepdim=True).values
                    concat_ = torch.cat((x_mean, x_max, x_min),dim=-1)
                    return_value[-1].append(concat_.cpu())

                else:
                    return_value[-1].append(
                    cache_dict_[el][:, extraction_tokens].cpu())
            else:
                return_value[-1].append(
                        cache_dict_[el][:, :, extraction_tokens].cpu())

        return_value[-1] = torch.stack(return_value[-1], dim=1)
    return_value = torch.stack(return_value, dim=1)
    return return_value
        

def extract_hidden_states(dataloader, tokenizer, model, logger,
                          extraction_layers=[0, 1],
                          extraction_locs=[1, 7],
                          extraction_tokens=[-1],
                          do_final_cat = True, return_tokenized_input = False,concat=False):
    assert [extraction_loc in extraction_locations.keys() for extraction_loc in extraction_locs]    
    assert (10 not in extraction_locs) or len(extraction_locs) == 1
        
    output_attentions = 10 in extraction_locs

    return_values = []
    
    tokenized_input = []
    
    for i, (batch_texts, _) in tqdm(enumerate(dataloader), total=len(dataloader)):

        inputs = tokenizer(
            batch_texts,
            padding='longest',
            truncation=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.run_with_cache(**inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions)

        cache_dict_ = outputs[1]

        r = extract_from_cache(cache_dict_, extraction_layers=extraction_layers,
                          extraction_locs=extraction_locs,
                          extraction_tokens=extraction_tokens,concat=concat)
        
        return_values.append(r)
        
        if return_tokenized_input:
            assert len(inputs['input_ids']) == 1, "Batch size must be 1 for tokenized input extraction"
            tokenized_input.append(tokenizer.convert_ids_to_tokens([w for w in inputs['input_ids'][0].cpu()]))

    if do_final_cat:
        return_values = torch.cat(return_values, dim=0)
    
    if return_tokenized_input:
        return return_values, tokenized_input
    return return_values


def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
