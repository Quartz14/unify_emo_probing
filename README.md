# A Unified View on Emotion Representation in Large Language Models

Code repository for paper "A Unified View on Emotion Representation in Large Language Models".

## Requirements
Please refer to environment.yml
```
conda env create -f environment.yml
conda activate emo_probe
```

## Datasets
The exact dataset splits used for the experiments is shared in the dataset folder. The datasets were obtained from [llm_data](https://github.com/andyzoujm/representation-engineering/tree/main/data/emotions), [crowd_event](https://github.com/aminbana/emo-llm/tree/main/data), [twitter](https://github.com/sisinflab/LLM-SentimentProber/blob/main/src/datasets/reduced_emotion_train.csv), [dailydialog](https://github.com/declare-lab/RECCON/tree/main/data/original_annotation). For the CoT experiment the [EmoBench](https://github.com/Sahandfer/EmoBench/blob/master/data/EU.jsonl) dataset was used. 

## Experiments
The primary code for Supervised and Unsupervised probing is in `main.py` in the respective folders. Code for generation and evaluation of the CoT experiments in the the CoT_expt folder.
The file can be run directly with `python main.py` passing the relevant arguments and right dataset paths. 
The respective folders include the codes for generating the plots in the paper.

The code for supervised_probing was adapted from [emo-llm](https://github.com/aminbana/emo-llm) and the code for unsupervised_probing was adapted from [representation-engineering](https://github.com/andyzoujm/representation-engineering). Gemini-2.5 Language Model was used to generate some of the code snippets for the plots. 

## Citation
Our paper is recently accepted at EACL 2026, official citation to be updated.
