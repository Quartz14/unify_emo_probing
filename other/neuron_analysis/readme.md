This directory consists of the code used for emotion neuron identification. It was adapted from [Language-Specific-Neurons-LAPE](https://github.com/RUCAIBox/Language-Specific-Neurons/tree/main). 

## To run
```
python activation_hf.py --data_type dialogue -gpu_id '0'
wait
python identify.py --data_type dialogue

[data_type = (llm, crowd, crowd-h, dialogue, twitter)]
```



The activation_hy.py and identify.py files included contribution from [Maharaj](https://github.com/maharajbrahma) to adapt the original code to support models loaded via HuggingFace. 
