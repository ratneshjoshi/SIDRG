import pandas as pd

from embedding_metrics import greedy_match_sentence_level, extrema_sentence_level, average_sentence_level
from embedding_metrics import greedy_match_corpus_level, extrema_corpus_level, average_corpus_level

from nltk.translate.bleu_score import sentence_bleu
from datasets import load_metric
from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk.translate.bleu_score import corpus_bleu

def evaluate(results):
    filename = str(results)+'.csv'

    print("evaluation for "+str(results)+" started")

    df = pd.read_csv(filename)

    Actual = [x.strip('[Bot]') for x in df['Actual']]
    Response = [str(x).strip('[SOR][Bot]') for x in df['Response']]

    bertscore_metric = load_metric("bertscore")

    sentence_bleu_list = []
    bert_score_list = []
    for x, y in zip(Actual, Response):
        sentence_bleu_list.append(str(sentence_bleu(x, y, weights=(1.0, 0, 0, 0))) + ":" + 
    	                      str(sentence_bleu(x, y, weights=(0.5, 0.5, 0, 0))) +":" + 
    	                      str(sentence_bleu(x, y, weights=(0.25, 0.25, 0.25, 0.25))))
        
        bert_score_list.append(bertscore_metric.compute(references=[x], predictions=[y], lang="en"))

    print("Bleu/bert done")

    file = "./Word2Vec/GoogleNews-vectors-negative300.bin.gz"

    embeddings = Word2VecKeyedVectors.load_word2vec_format(file, binary=True)

    greedy_list = []
    extrema_list = []
    average_list = []

    for x, y in zip(Actual, Response):
        greedy_list.append(greedy_match_sentence_level(x, y, embeddings))
        extrema_list.append(extrema_sentence_level(x, y, embeddings))
        average_list.append(average_sentence_level(x, y, embeddings))
        
    print("embedding based done")
    print("sentence level done")
      
    bleu1_score = corpus_bleu(Actual, Response, weights=(1.0, 0, 0, 0))
    bleu2_score = corpus_bleu(Actual, Response, weights=(0.5, 0.5, 0, 0))
    bleu4_score = corpus_bleu(Actual, Response, weights=(0.25, 0.25, 0.25, 0.25))
    bert_score = bertscore_metric.compute(references=[Actual], predictions=[Response], lang="en")
    print("Bleu/bert done")

    greedy = greedy_match_corpus_level(Actual, Response, embeddings)
    extrema = extrema_corpus_level(Actual, Response, embeddings)
    average = average_corpus_level(Actual, Response, embeddings)
    print("embedding based done")
    print("corpus level done")

    print("Bleu1 score is ", bleu1_score)
    print("Bleu2 score is ", bleu2_score)
    print("Bleu4 score is ", bleu4_score)
    print("Bert score is ", bert_score)
    print("Greedy embedding match is ", greedy)
    print("Extrema embedding match is ", extrema)
    print("Average embedding match is ", average)

    with open('evaluation_coupus.txt', 'a') as file:
        file.write("evaluation for "+str(results)+'\n')
        file.write("Bleu1 score is "+str(bleu1_score)+'\n'+
                    "Bleu2 score is "+str(bleu2_score)+'\n'+
                    "Bleu4 score is "+str(bleu4_score)+'\n'+
                    "Bert score is "+str(bert_score)+'\n'+
                    "Greedy embedding match is "+str(greedy)+'\n'+
                    "Extrema embedding match is "+str(extrema)+'\n'+
                    "Average embedding match is "+str(average)+'\n\n')

    df['bleu'] = sentence_bleu_list
    df['bertscore'] = bert_score_list
    df['greedy'] = greedy_list
    df['extrema'] = extrema_list
    df['average'] = average_list

    savefile = str(results)+'-eval.csv'
    df.to_csv(savefile, index=False)


# result_list = ['results-base', 'results-dialog_acts', 'results-slots', 'results-belief', 'results-keywords', 'results-kg']

result_list = ['results-base', 'results-dialog_acts', 'results-keywords', 'results-kg']

# result_list = ['results-kg']


for results in result_list:
    evaluate(results)


