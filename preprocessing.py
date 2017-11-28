import json

import numpy as np
import pandas as pd
import gensim
from gensim.models.word2vec import Word2Vec  # the word2vec model gensim class
from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tokenizer = TweetTokenizer()
LabeledSentence = gensim.models.doc2vec.LabeledSentence  # we'll talk about this down below
W2V_N_DIM = 150
W2V_MODEL_NAME = 'tweet_w2v_model.json'

FILENAME_A = 'data/semeval_train_A_merged.txt'
FILENAME_B = 'data/semeval_train_B.txt'
FILENAME_C = 'data/semeval_train_C.txt'
BASE_PATH = './W2V_MODEL/'

INDEX_A = ['SentimentText', 'Sentiment']
INDEX_B = ['SentimentText', 'topic', 'Sentiment']
INDEx_C = ['SentimentText', 'topic', 'Sentiment']


def ingest(filename, index_names, sep='\t', encode='utf-8'):
    df = pd.read_csv(filename, sep=sep, header=None, names=index_names, encoding=encode)
    """
    print("============================== Preview ==============================")
    print(df.head(100))
    print("============================== Summary ==============================")
    print(df.describe())
    print("=====================================================================")
    """
    return df


def preprocess_tweet(data, n=-1):
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
    for i, v in tqdm(enumerate(tweets)):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


def train_w2v(train_data):
    tweet_w2v = Word2Vec(size=W2V_N_DIM, min_count=10)
    tweet_w2v.build_vocab([x.words for x in tqdm(train_data)])
    tweet_w2v.train([x.words for x in tqdm(train_data)], total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)
    return tweet_w2v


def save_w2v(model, filename):
    model.save(BASE_PATH + filename)


def load_w2v(filename):
    model = gensim.models.Word2Vec.load(BASE_PATH + filename)
    return model


def get_vocab_list(model):
    corpus = list(model.wv.vocab.keys())
    return corpus


def vocab2vec(model, vocabs):
    vocab_vector_dict = dict()
    for vocab in tqdm(vocabs):
        vector = model[vocab]
        vocab_vector_dict[vocab] = vector.tolist()

    print("Vocabulary Set converted to Vector [total: {} words]".format(len(vocab_vector_dict)))
    return vocab_vector_dict


def save_json(data, filename):
    with open(BASE_PATH + filename, 'w') as outfile:
        json.dump(data, outfile)


def load_json(filename):
    data = None
    with open(BASE_PATH + filename, 'r') as json_data:
        data = json.load(json_data)
    return data

def increase_dataset(filename):
    data_more = ingest(filename='./data/train_more.csv', index_names=['Sentiment', 'num', 'date', 'topic', 'user', 'SentimentText'], sep=',', encode='latin-1')
    data_more = data_more.drop('num', axis=1)
    data_more = data_more.drop('date', axis=1)
    data_more = data_more.drop('topic', axis=1)
    data_more = data_more.drop('user', axis=1)
    data_more.loc[data_more.Sentiment == 0, 'Sentiment'] = -1
    data_more.loc[data_more.Sentiment == 4, 'Sentiment'] = 1

    print(data_more.head(20))
    print(data_more.describe())

    data = ingest(filename=FILENAME_A, index_names=INDEX_A)
    data.loc[data.Sentiment == 'positive', 'Sentiment'] = 1
    data.loc[data.Sentiment == 'neutral', 'Sentiment'] = 0
    data.loc[data.Sentiment == 'negative', 'Sentiment'] = -1
    print(data.head(20))
    print(data.describe())

    merged_data = pd.concat([data_more, data])
    print(merged_data.head(10))
    print(merged_data.describe())

    merged_data = merged_data.drop_duplicates(subset='SentimentText', keep='last')
    print(merged_data.head(10))
    print(merged_data.describe())

    merged_data.to_csv(filename, sep='\t', encoding='utf-8', index=False, header=False)
    print("data saved")



if __name__ == '__main__':
    data = ingest(filename=FILENAME_A, index_names=INDEX_A)
    print("loadding done")

    # Loading txt file and preprocessing
    data = preprocess_tweet(data)
    print("preprocess done")
    # Devide train set and test set but now set test size to 0
    x_train, x_test, y_train, y_test = train_test_split(np.array(data.tokens),
                                                        np.array(data.Sentiment),
                                                        test_size=0.0)
    # Tag training data and test data
    x_train = labelizeTweets(x_train, 'TRAIN')
    x_test = labelizeTweets(x_test, 'TEST')

    # train and save word2vec model
    model = train_w2v(train_data=x_train)
    print("train done")
    save_w2v(model=model, filename=W2V_MODEL_NAME)
    print("save model done")
    # show how to load word2vec model
    w2v_model = load_w2v(filename=W2V_MODEL_NAME)
    print("load model done")
    # make vocabulary list from model
    corpus = get_vocab_list(w2v_model)

    # convert each vocabulary word to voctor using model
    vector_dict = vocab2vec(w2v_model, corpus)

    # save vocab-vector dictionary into json format
    save_json(vector_dict, "vocab_vector.json")

    # show how to load vocab-vector dictionary
    # vector size is 150 dimension now but it could be changed
    vocab_vector = load_json("vocab_vector.json")
    vocab_list = list(vocab_vector.keys())
    print(vocab_vector['you'])
    print(vocab_list[:10])

