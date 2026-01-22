import argparse
from types import MethodType
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("-d", "--data_type", type=str, help="[twitter, crowd-h, crowd, llm, dialogue]")
parser.add_argument('-gpu_id',type=str, default='0') 
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()


# To make only GPU device 0 visible
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

is_llama = True#"llama" in args.model.lower()
data_type = args.data_type

# get emotion list of each dataset
if(data_type=='twitter'):
    emotions_list = ["anger", "love","fear", "joy","sadness", "surprise"]
elif(data_type=='crowd'):
    emotions_list = ['anger','disgust', 'fear', 'joy', 'sadness', 'surprise']
elif(data_type=='crowd-h'):
    emotions_list = ['anger','disgust', 'fear', 'joy', 'sadness', 'surprise']
elif(data_type=='llm'):
    emotions_list = ['anger','disgust','fear','happiness','sadness','surprise']
elif(data_type=='dialogue'):
    emotions_list = ['anger','disgust', 'fear', 'joy', 'sadness', 'surprise']


print(f"[INFO] Loading model: {args.model}")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    # torch_dtype=torch.bfloat16,  # Best for A100
    device_map="auto"
)
model.eval()
print(f"[INFO] CUDA memory after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


num_layers = model.config.num_hidden_layers
intermediate_size = (
    model.config.intermediate_size #if is_llama else model.config.hidden_size * 4
)
over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to("cuda")


def factory(idx):
    if is_llama:
        def llama_forward(self, x):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            gate = torch.nn.functional.silu(gate)
            activation = gate.float()
            over_zero[idx, :] += (activation > 0).sum(dim=(0, 1))
            x = gate * up
            x = self.down_proj(x)
            return x
        return llama_forward
    else:
        def bloom_forward(self, x):
            x = self.dense_h_to_4h(x)
            x = self.gelu_impl(x)
            activation = x.float()
            over_zero[idx, :] += (activation > 0).sum(dim=(0, 1))
            x = self.dense_4h_to_h(x)
            return x
        return bloom_forward

for i in range(num_layers):
    if is_llama:
        obj = model.model.layers[i].mlp
    else:
        obj = model.transformer.h[i].mlp
    obj.forward = MethodType(factory(i), obj)

print(f"[INFO] Patched {num_layers} MLP layers for activation counting.")


for emotion in emotions_list:
    over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to("cuda")

    # data_path = f"data/emotions/{data_type}/{emotion}.train.llama"
    data_path = f"data/emotions_inst/{data_type}/{emotion}.train.llama"
    print(f"[INFO] Loading token IDs from {data_path}")
    ids = torch.load(data_path, weights_only=False)
    print(f"[INFO] Loaded {ids.numel():,} tokens")

    input_ids = ids.to(model.device)
    l = 0
    l_sanity = 0
    print(f"[INFO] Using {input_ids.size(0)} sequences of MAX length {input_ids.size(1)}")
    print(f"[INFO] CUDA memory before inference: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    batch_size = args.batch_size
    print(f"[INFO] Starting activation counting (batch_size={batch_size})...")

    with torch.no_grad():
        for i in tqdm(range(0, input_ids.size(0), batch_size)):
            batch = input_ids[i:i+batch_size]
            if(i==0):
                print(batch.shape)
            batch = batch[batch != tokenizer.eos_token_id]
            
            batch = batch.unsqueeze(0)
            l += len(batch[0])
            if(i==0):
                print(batch.shape)
            _ = model(batch)
            l_sanity += batch.shape[-1]
            torch.cuda.synchronize()

    output = dict(n=l, over_zero=over_zero.to("cpu"))
    out_path = f"data/emotions_inst/{data_type}/activation.{emotion}.train.llama-8b" 
    torch.save(output, out_path)
    print(f"Saved activation of {data_type}/{emotion} stats to {out_path}")
    print("Total tokens processed = ",l)
    print("Total tokens processed sanity= ",l_sanity)

print(f"done with saving activations of {data_type}")
