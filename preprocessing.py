import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
color = sns.color_palette()

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from pprint import pprint

tokenizer = TweetTokenizer()
LabeledSentence = gensim.models.doc2vec.LabeledSentence  # we'll talk about this down below
W2V_N_DIM = 150
W2V_MODEL_NAME = 'tweet_w2v_model.json'

FILENAME_A = 'data/semeval_train_A.txt'
FILENAME_B = 'data/semeval_train_B.txt'
FILENAME_C = 'data/semeval_train_C.txt'

INDEX_A = ['SentimentText', 'Sentiment']
INDEX_B = ['ttext', 'topic', 'sentiment']
INDEx_C = ['ttext', 'topic', 'point']


def ingest(filename, index_names):
    df = pd.read_csv(filename, sep="\t", header=None, names=index_names)
    print("============================== Preview ==============================")
    print(df.head(5))
    print("============================== Summary ==============================")
    print(df.describe())
    print("=====================================================================")
    return df

def postprocess(data, n=-1):

    def tokenize(tweet):
        tweet = tweet['SentimentText']
        try:
            tweet = tweet.lower()  # unicode(tweet.decode('utf-8').lower())
            tokens = tokenizer.tokenize(tweet)
            tokens = filter(lambda t: not t.startswith('@'), tokens)
            tokens = filter(lambda t: not t.startswith('#'), tokens)
            tokens = filter(lambda t: not t.startswith('http'), tokens)
            tokens = list(tokens)
            return tokens
        except:
            return 'NC'

    if n > 0:
        data = data.head(n)
    data['tokens'] = data.apply(lambda row: tokenize(row), axis=1)
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

def train_w2v(train_data):
    tweet_w2v = Word2Vec(size=W2V_N_DIM, min_count=10)
    tweet_w2v.build_vocab([x.words for x in tqdm(train_data)])
    tweet_w2v.train([x.words for x in tqdm(train_data)], total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)
    return tweet_w2v

def save_w2v(model, filename):
    model.save(filename)

def load_w2v(filename):
    model = gensim.models.Word2Vec.load(filename)
    return model

def get_vocab_list(model):
    corpus = list(model.wv.vocab.keys())
    return corpus

def vocab2vec(model, vocabs):
    vocab_vector_dict = dict()
    for vocab in tqdm(vocabs):
        vector = model[vocab]
        vocab_vector_dict[vocab] = vector.tolist()
    return vocab_vector_dict

def save_json(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)




if __name__ == '__main__':


    data = ingest(filename=FILENAME_A, index_names=INDEX_A)
    data = postprocess(data)

    x_train, x_test, y_train, y_test = train_test_split(np.array(data.tokens),
                                                        np.array(data.Sentiment),
                                                        test_size=0.0)

    x_train = labelizeTweets(x_train, 'TRAIN')
    x_test = labelizeTweets(x_test, 'TEST')

    # model = train_w2v(train_data=x_train)
    # save_w2v(model=model, filename=W2V_MODEL_NAME)
    w2v_model = load_w2v(filename=W2V_MODEL_NAME)
    corpus = get_vocab_list(w2v_model)
    vector_dict = vocab2vec(w2v_model, corpus)
    print("Vocabulary Set converted to Vector [total: {} words]".format(len(vector_dict)))
    save_json(vector_dict, "vocab_vector.json")
    # voca = tweet_w2v.sorted_vocab



    # a = 'dec 21st 2012 will be know not as the end'
    # print(tokenize(a))
