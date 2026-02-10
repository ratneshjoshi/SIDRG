import json
import pandas as pd


df_train = pd.read_csv('train.csv')
df_train = df_train[['utterances']]
words = []
for i in df_train['utterances'].to_list():
    for j in i.split(' '):
        if j not in words:
            words.append(j)


from keybert import KeyBERT
kw_model = KeyBERT()

def get_keyBert_keywords(text):
    return kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english')

with open('concept/cc_kg.json', encoding='utf-8') as file:
    data = json.load(file)

def return_keywords(text):
    text_keyword = get_keyBert_keywords(text)
    final_keyword = []
    for keyword in text_keyword:
        final_keyword.append(keyword[0])
    
    final_keyword = str(' '.join(final_keyword))
    return final_keyword

def return_triples(keys):
    trip = []
    for key in keys:
        try:
            for x in data[key]:
                if x[1] in ['FormOf', 'IsA', 'Causes', 'SimilarTo', 'HasA'] and x[2] in words:
                    trip.append(' '.join(x))
        except:
            pass
    trip = list(set(trip))
    trip = ' '.join(trip)
    return trip


conv_id = []
turn_no = []
speaker = []
utterances = []
act = []
emotion = []
keyword = []
triples = []



with open("data/ijcnlp_dailydialog/train/dialogues_act_train.txt", 'r', encoding="utf8") as act_file:
    with open("data/ijcnlp_dailydialog/train/dialogues_emotion_train.txt", 'r', encoding="utf8") as emotion_file:
        with open("data/ijcnlp_dailydialog/train/dialogues_train.txt", 'r', encoding="utf8") as data_file:

            data_list = data_file.read().split('\n')
            emotion_list = emotion_file.read().split('\n')
            act_list = act_file.read().split('\n')

            act_dict = { 1: 'inform', 2: 'question', 3: 'directive', 4: 'commissive' }
            emotion_dict = { 0: 'no_emotion', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'happiness', 5: 'sadness', 6: 'surprise'}

            for ids, (dialog, emotions, acts) in enumerate(zip(data_list, emotion_list, act_list)):

                dialog = dialog.split('__eou__')
                emotions = emotions.split(' ')
                acts = acts.split(' ')
                
                for i in range(len(dialog)-1):
                    conv_id.append(ids)
                    turn_no.append(i)
                    if i%2==0:
                        speaker.append('USER')
                    else:
                        speaker.append('BOT')
                    utterances.append(dialog[i])
                    act.append(act_dict[int(acts[i])])
                    emotion.append(emotion_dict[int(emotions[i])])
                    keywords = return_keywords(dialog[i])
                    keyword.append(keywords)
                    if keywords == '':
                        triples.append('')
                    else:
                        triples.append(return_triples(keywords.split(' ')))

                if ids==0:
                    print(keyword[-1])
                    print(triples[-1])

                if (ids+1)%10==0:
                    print(ids+1, 'dialogues done out of ', len(data_list))

df = pd.DataFrame.from_dict({
                                'conv_id':conv_id, 
                                'turn_no':turn_no,
                                'speaker':speaker,
                                'utterances':utterances,
                                'dialog_act':act,
                                'emotion':emotion,
                                'keywords':keyword,
                                'triples':triples
                                })

df.to_csv('train.csv', index=False)

