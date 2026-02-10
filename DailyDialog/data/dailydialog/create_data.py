import pandas as pd


df_eval = pd.read_csv('val.csv')
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')

df_train = df_train[['turn_no', 'utterances', 'speaker','dialog_act',  'keywords', 'triples']]
df_test = df_test[['turn_no', 'utterances', 'speaker', 'dialog_act',  'keywords', 'triples']]
df_eval = df_eval[['turn_no', 'utterances', 'speaker', 'dialog_act', 'keywords', 'triples']]


def format_data(df, filename):
    data = []
    for i, row in df.iterrows():
        if row['speaker']=='BOT' and row['turn_no']>2:

            instance = '[SOC]' + '[USER]' + str(df.iloc[i-3]['utterances']) + \
                        '[SYSTEM]' + str(df.iloc[i-2]['utterances']) + \
                        '[USER]' + str(df.iloc[i-1]['utterances']) + '[EOC]' + \
                        '[SOR][SYSTEM]' + str(df.iloc[i]['utterances']) + '[EOR]'
            data.append(instance)
        # if (i+1)%100==0:
        #     print(i+1, "out of", len(df), "done")
    print(str(filename) + " Done")

    df_new = pd.DataFrame.from_dict({'text':data})
    df_new.to_csv(filename, index=False)

format_data(df_train, 'data/base/train.csv')
format_data(df_eval, 'data/base/eval.csv')
format_data(df_test, 'data/base/test.csv')


def format_data_act(df, filename):
    data = []
    for i, row in df.iterrows():
        if row['speaker']=='BOT' and row['turn_no']>2:

            instance = '[SOC]' + '[USER]' + df.iloc[i-3]['utterances'] + '[dialog_acts]' + df.iloc[i-3]['dialog_act'] + \
                        '[SYSTEM]' + df.iloc[i-2]['utterances'] + '[dialog_acts]' + df.iloc[i-2]['dialog_act'] + \
                        '[USER]' + df.iloc[i-1]['utterances'] + '[dialog_acts]' + df.iloc[i-1]['dialog_act'] + '[EOC]' + \
                        '[SOR][SYSTEM]' + df.iloc[i]['utterances'] + '[EOR]'
            data.append(instance)
        # if (i+1)%100==0:
        #     print(i+1, "out of", len(df), "done")
    print(str(filename) + " Done")

    df_new = pd.DataFrame.from_dict({'text':data})
    df_new.to_csv(filename, index=False)

format_data_act(df_train, 'data/dialog_acts/train.csv')
format_data_act(df_eval, 'data/dialog_acts/eval.csv')
format_data_act(df_test, 'data/dialog_acts/test.csv')



def format_data_keyword(df, filename):
    data = []
    for i, row in df.iterrows():
        if row['speaker']=='BOT' and row['turn_no']>2:

            instance = '[SOC]' + '[USER]' + str(df.iloc[i-3]['utterances']) + '[keywords]' + str(df.iloc[i-3]['keywords']) + \
                        '[SYSTEM]' + str(df.iloc[i-2]['utterances'] )+ '[keywords]' + str(df.iloc[i-2]['keywords']) + \
                        '[USER]' + str(df.iloc[i-1]['utterances']) + '[keywords]' + str(df.iloc[i-1]['keywords']) + '[EOC]' + \
                        '[SOR][SYSTEM]' + str(df.iloc[i]['utterances']) + '[EOR]'
            data.append(instance)
        # if (i+1)%100==0:
        #     print(i+1, "out of", len(df), "done")
    print(str(filename) + " Done")

    df_new = pd.DataFrame.from_dict({'text':data})
    df_new.to_csv(filename, index=False)

format_data_keyword(df_train, 'data/keywords/train.csv')
format_data_keyword(df_eval, 'data/keywords/eval.csv')
format_data_keyword(df_test, 'data/keywords/test.csv')


def format_data_triples(df, filename):
    data = []
    for i, row in df.iterrows():
        if row['speaker']=='BOT' and row['turn_no']>2:

            instance = '[SOC]' + '[USER]' + str(df.iloc[i-3]['utterances']) + '[triples]' + str(df.iloc[i-3]['triples']) + \
                        '[SYSTEM]' + str(df.iloc[i-2]['utterances']) + '[triples]' + str(df.iloc[i-2]['triples']) + \
                        '[USER]' + str(df.iloc[i-1]['utterances']) + '[triples]' + str(df.iloc[i-1]['triples']) + '[EOC]' + \
                        '[SOR][SYSTEM]' + str(df.iloc[i]['utterances']) + '[EOR]'
            data.append(instance)
        # if (i+1)%100==0:
        #     print(i+1, "out of", len(df), "done")
    print(str(filename) + " Done")

    df_new = pd.DataFrame.from_dict({'text':data})
    df_new.to_csv(filename, index=False)

format_data_triples(df_train, 'data/kg/train.csv')
format_data_triples(df_eval, 'data/kg/eval.csv')
format_data_triples(df_test, 'data/kg/test.csv')