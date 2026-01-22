def build_prompt(shots = ('joy', 'sadness'), prompt_index = 0, version='P2'):
    sample_shots = [
        ('My husband missed an important call, because his phone was on silent AGAIN!','anger'),
        ('At work, my job is far to mind numbing and it does not challenge me.', 'boredom'),
        ('I saw mouldy food.', 'disgust'),
        ('I was confronted by a thief.', 'fear'),
        ('I booked a night away with friends and my children were upset that I was going to a hotel without them.', 'guilt'),
        ('My first child was born.', 'joy'),
        ('I walked to the village near my home.', 'neutral'),
        ('I was recognised for doing well at work by my line manager.', 'pride'),
        ("I found out my mam didn't need a serious surgery.", 'relief'),
        ('My dog died last week.', 'sadness'),
        ('I failed a maths test and the teacher called me out in front of the class.', 'shame'),
        ('I saw my teacher on an aeroplane!', 'surprise'),
        ('I gave my wallet to a friend.', 'trust'),
    ]
    sample_shots_dict = {emotion: shot for shot, emotion in sample_shots}
    sample_shots = [(sample_shots_dict[shot], shot) for shot in shots]
    
    emotion_list = "Consider this list of emotions: anger, boredom, disgust, fear, guilt, joy, pride, relief, sadness, shame, surprise, trust, neutral. "
    

    prompt0 = f"What are the inferred emotions in the following contexts?"
    prompt1 = emotion_list + prompt0
    prompt2 = ''
    prompt3 = 'Guess the emotion.'
    prompt4 = f'Decipher the emotion from the following statements: '
    prompt5 = f'Decipher the label for the following statements: '
    prompt6 = f'What is the label, for the statement? '
    prompt7 = f'What is the label, given the context? '
    promot8 = emotion_list + prompt4
    promot9 = emotion_list + prompt5

    


    prompt = [prompt0, prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7][prompt_index]
    #
    for shot, emotion in sample_shots:
        prompt += f" Context: {shot} Answer: {emotion}"
    func = lambda x: f'{prompt} Context: {x} Answer:'

    if(version=="P2-d"):
        func = lambda x: f'What is the inferred emotion in the following context? Context: A: Today was a memorable day. B: Oh, what happened? A: My first child was born. Emotion of A: joy Context: B: I have been feeling low lately. A: Do you mind sharing what has been bothering you lately? B: My dog died last week. Emotion of B: sadness Context: {x}'
        return func
    

    if(version=='P1'):
        func = lambda x: f'What is the inferred emotion in the following context? Context: {x} Answer:'
        return func

    if(version=='P1_long'):
        func = lambda x: f"Please infer the primary emotion that is being communicated or suggested within the following context. Answer based on the overall tone and expressed feelings present in the text as a whole. Context: {x}"
        return func
    
    
    if(version=='P0'):
        func = lambda x: f'{x}'
        return func

    return func

if __name__ == '__main__':
    prompt = build_prompt(shots = ('fear', 'anger', 'sadness'), prompt_index = 7)
    print(prompt('I was happy to see my friend.'))
