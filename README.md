# A Unified View on Emotion Representation in Large Language Models

Code repository for paper "A Unified View on Emotion Representation in Large Language Models".

## Motivation
Although several works have used interpretability techniques to understand emotion identification capabilities of LLMs, there are contradictions in the findings regarding the layers early / middle where emotion representation exists. We opine that these contradictions arise due to the use of different prompts and datasets having different difficulty levels. Our intent is to analyze the setup using prompts with different expressivity levels across datasets with different difficulty levels – to get an overall understanding of the ability of LLMs in identifying emotions at their representation level. We additionally try to assess the reasoning ability of such models with an emotion comprehension task.

## Key Findings
* Using probing techniques we show the layers with emotion representation depends on influence of the instruction prompt and the clarity with which emotion is expressed in the input data.
* There exists intrinsic emotion reading vectors that are similar across datasets and can be used interchangeably, revealing their foundational nature.
* LLMs performance on emotion reasoning tasks remains poor, we observe CoT mostly generates reasoning traces to increase it confidence in its original answer, especially when the model is confident in it.
* This motivates the need for methods that can **leverage the implicit emotion representations to improve explicit reasoning capabilities of LLMs**.

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
Our paper is recently accepted at EACL 2026, 
If you find our work useful for your research, please kindly cite our paper as follows (official citation to be updated).
