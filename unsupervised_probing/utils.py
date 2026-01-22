import json
import numpy as np
import random
import os
import pandas as pd
import re

def primary_emotions_concept_dataset(case,data_dir="datasets/rep_llm_data", user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    if (case=='explicit_emotion'):
        template_str = 'Consider the {emotion} of the following scenario:\nScenario: {scenario}\nAnswer: '
    elif(case == 'no_emotion'):
        template_str = 'What is the inferred emotion in the following scenario:\nScenario: {scenario}\nAnswer: '
    elif(case == 'dialogue'):
        template_str = 'What is the inferred emotion in the following context? Speaker {speaker}: {scenario} Emotion of {speaker}: Answer: '

    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    raw_data = {}
    for emotion in emotions:
        with open(os.path.join(data_dir, f'{emotion}.json')) as file:
            # raw_data[emotion] = json.load(file)
            raw_data[emotion] = list(set(json.load(file)))[:200]

    formatted_data = {}
    for emotion in emotions:
        c_e, o_e= raw_data[emotion], np.concatenate([v for k,v in raw_data.items() if k != emotion])
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        train_labels2 = []

        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        
        data = np.concatenate(data).tolist()
        speaker_list_data = [random.choice(['A','B']) for i in range(len(data)//2)]
        speaker_list_data.extend(speaker_list_data)

        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        speaker_list_data_ = [random.choice(['A','B']) for i in range(len(data_)//2)]
        speaker_list_data_.extend(speaker_list_data_)
 
        emotion_test_data = [template_str.format(emotion=emotion, scenario=d, speaker=s).strip() for d,s in zip(data_, speaker_list_data_)]
        emotion_train_data = [template_str.format(emotion=emotion, scenario=d, speaker=s).strip() for d,s in zip(data, speaker_list_data)]
        

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels, 'labels_raw':train_labels2},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data


def primary_emotions_concept_dataset_crowd(hidden,data_dir="datasets/crowd-event_train.csv"):
    random.seed(0)
    df = pd.read_csv(data_dir)
    template_str = 'What is the inferred emotion in the following scenario:\nScenario: {scenario}\nAnswer: '
    emotions = ["joy", "sadness", "anger", "fear", "disgust", "surprise"]
    raw_data = {}
    for emotion in emotions:
        df_emo = df[df['emotion']==emotion]
        # random sample 200
        df_emo = df_emo.sample(n=200)
        if(hidden):
            raw_data[emotion] = list(df_emo['hidden_emo_text'])
        else:
            raw_data[emotion] = list(df_emo['text'])

    formatted_data = {}
    for emotion in emotions:
        c_e, o_e= raw_data[emotion], np.concatenate([v for k,v in raw_data.items() if k != emotion])
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        
        data = np.concatenate(data).tolist()
        speaker_list_data = [random.choice(['A','B']) for i in range(len(data)//2)]
        speaker_list_data.extend(speaker_list_data)

        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        speaker_list_data_ = [random.choice(['A','B']) for i in range(len(data_)//2)]
        speaker_list_data_.extend(speaker_list_data_)
        
        emotion_test_data = [template_str.format(emotion=emotion, scenario=d).strip() for d in data_]
        emotion_train_data = [template_str.format(emotion=emotion, scenario=d).strip() for d in data]

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data


def primary_emotions_concept_dataset_twitter(data_dir="datasets/reduced_emotion_train.csv"):
    random.seed(0)
    df = pd.read_csv(data_dir)
    template_str = 'What is the inferred emotion in the following scenario:\nScenario: {scenario}\nAnswer: '
    emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    emotion_to_id =  {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
    id_to_emotion =  {0:"sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    raw_data = {}
    df['emotion'] = [id_to_emotion[i] for i in list(df['label'])]
    for emotion in emotions:
        df_emo = df[df['emotion']==emotion]
        # random sample 200
        df_emo = df_emo.sample(n=200)
        raw_data[emotion] = list(df_emo['text'])

    formatted_data = {}
    for emotion in emotions:
        c_e, o_e= raw_data[emotion], np.concatenate([v for k,v in raw_data.items() if k != emotion])
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        
        data = np.concatenate(data).tolist()
        speaker_list_data = [random.choice(['A','B']) for i in range(len(data)//2)]
        speaker_list_data.extend(speaker_list_data)

        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        speaker_list_data_ = [random.choice(['A','B']) for i in range(len(data_)//2)]
        speaker_list_data_.extend(speaker_list_data_)
        
        emotion_test_data = [template_str.format(emotion=emotion, scenario=d).strip() for d in data_]
        emotion_train_data = [template_str.format(emotion=emotion, scenario=d).strip() for d in data]

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data


def primary_emotions_concept_dataset_dialogue(emotion_label='last',conv_type='full',data_dir="datasets/dailydialog_train_processed_filtered.json", user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    df = pd.read_csv(data_dir)
    df['conversation_raw'] = [x.strip() for x in list(df['conversation'])] # For whole conv
    if(emotion_label=='prev_emotion'):
        # df['conversation_raw'] = [x.strip() for x in list(df['conversation_prev'])]
        df['conversation_raw'] = [x.strip() for x in list(df['conversation_prev_rep'])]

    last_utterance = []
    for i in range(len(df)):
        tmp = df['conversation'][i].split(":")
        tmp_target = [i for i in tmp if "Emotion of" in i]
        last_utterance.append(tmp_target[0].split("Emotion of")[0].strip())
        # last_utterance.append(tmp_target[0].strip())
    df['last_utterance'] = last_utterance # for last utterance 
    if(emotion_label=='last'):
        emotion_label = 'emotion'
    elif(emotion_label=='prev'):
        emotion_label = 'prev_emotion'
    

    # template_str = '{user_tag} Consider the {emotion} of the following scenario:\nScenario: {scenario}\nAnswer: {assistant_tag} '
    # template_str = '{user_tag} Consider the emotion {emotion} expressed in the following conversation snippet:\nConversation: {scenario}\nemotion: {assistant_tag} '
    # template_str = '{user_tag} Consider the emotion {emotion} expressed in the following conversation snippet:\nConversation: {scenario}\nAnswer: {assistant_tag} '
    # template_str = 'What is the inferred emotion in the following context? Conversation: {x} Answer:'
    template_str = 'What is the inferred emotion in the following scenario:\nScenario: {x}\nAnswer: '
    # template_str = 'What is the inferred emotion of the second last speaker in the following context? Conversation: {x} Answer:'
    # template_str = 'What is the inferred emotion in the following context? Speaker {speaker}: {scenario} Emotion of {speaker}: Answer: '

    emotions = ["joy", "sadness", "anger", "fear", "disgust", "surprise"]
    raw_data = {}
    for emotion in emotions:
        if(conv_type=='full'):
            raw_data[emotion] = list(set(df[df[emotion_label]==emotion]['conversation_raw']))#[:200]  For whole conv
            # raw_data[emotion] = list(set(df[df['emotion']==emotion]['conversation']))#[:200]  For whole conv
        elif(conv_type=='last'):
            raw_data[emotion] = list(set(df[df[emotion_label]==emotion]['last_utterance']))#[:200]  For last utterance

    formatted_data = {}
    for emotion in emotions:
        c_e, o_e= raw_data[emotion], np.concatenate([v for k,v in raw_data.items() if k != emotion])
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        train_labels2 = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        data = np.concatenate(data).tolist()
        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()

        emotion_test_data = [template_str.format(x=d).strip() for d in data_]
        emotion_train_data = [template_str.format(x=d).strip() for d in data]
        

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels, 'labels_raw':train_labels2},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data
