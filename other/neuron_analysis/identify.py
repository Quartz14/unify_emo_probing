import torch
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_type", type=str, help="[twitter, crowd-h, crowd, llm, dialogue]")
args = parser.parse_args()

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


n, over_zero = [], []


for emotion in emotions_list:
    
    data = torch.load(f'data/emotions_inst/{data_type}/activation.{emotion}.train.llama-8b')
    # data = torch.load(f'data/emotions/{data_type}/activation.{emotion}.train.llama-8b')
    n.append(data['n'])
    over_zero.append(data['over_zero'])

n = torch.tensor(n)
print("n - total tokens (denominator) : ", n)
over_zero = torch.stack(over_zero, dim=-1)
# n = over_zero.shape[0]*over_zero.shape[1]*over_zero.shape[2]#layer x inter x lang_num
print("over_zero shape = ", over_zero.shape)
num_layers, intermediate_size, lang_num = over_zero.size()



def activation():
    top_rate = 0.01
    filter_rate = 0.95
    activation_bar_ratio = 0.95
    activation_probs = over_zero / n # layer x inter x lang_num
    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    largest = False
    
    if torch.isnan(entropy).sum():
        print(torch.isnan(entropy).sum())
        raise ValueError
    
    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * filter_rate)).values.item()
    print(top_prob_value)
    # dismiss the neruon if no language has an activation value over top 90%
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    flattened_entropy = entropy.flatten()
    top_entropy_value = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index] # n x lang
    # for r, c in zip(row_index, col_index):
    #     print(r, c, activation_probs[r][c])

    print(selected_probs.size(0), torch.bincount(selected_probs.argmax(dim=-1)))
    selected_probs = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * activation_bar_ratio)).values.item()
    print((selected_probs > activation_bar).sum(dim=1).tolist())
    lang, indice = torch.where(selected_probs > activation_bar)

    merged_index = torch.stack((row_index, col_index), dim=-1)
    final_indice = []
    for _, index in enumerate(indice.split(torch.bincount(lang).tolist())):
        lang_index = [tuple(row.tolist()) for row in merged_index[index]]
        lang_index.sort()
        layer_index = [[] for _ in range(num_layers)]
        for l, h in lang_index:
            layer_index[l].append(h)
        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()
        final_indice.append(layer_index)
    # torch.save(final_indice, f"activation_mask/{data_type}_llama-8b")  
    # torch.save(final_indice, f"activation_mask/llama_inst/{data_type}_llama-8b")  
    torch.save(final_indice, f"activation_mask/gemma_inst/{data_type}_gemma-9b")  

activation()


print("Emotion specific neurons for ",data_type)
identify_data = torch.load(f"activation_mask/gemma_inst/{data_type}_gemma-9b", weights_only= False)
# identify_data = torch.load(f"activation_mask/llama_inst/{data_type}_llama-8b", weights_only= False)
# identify_data = torch.load(f"activation_mask/{data_type}_llama-8b", weights_only= False)
for i,emotion in enumerate(emotions_list):
    print(data_type, emotion, [len(ij) for ij in identify_data[i]])

print("________________________________________________________")
print("________________________________________________________")
