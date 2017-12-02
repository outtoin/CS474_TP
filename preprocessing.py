import json

import numpy as np
import pandas as pd
from gensim import models
from gensim.models.word2vec import Word2Vec  # the word2vec model gensim class
from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from spellchecker import SpellChecker

tokenizer = TweetTokenizer()
ps = PorterStemmer()

stop_words = set(stopwords.words("english"))
stop_words.update([',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])  # remove it if you need punctuation
whitelist = ["n't", "not", "hadn", "didn", "did",
             "no", "but", "wasn", "mustn", "was",
             "doesn", "aren", "can", "nor", "hasn",
             "does", "should", "shouldn"]
for white_word in whitelist:
    if white_word in stop_words:
        stop_words.remove(white_word)

LabeledSentence = models.doc2vec.LabeledSentence
W2V_N_DIM = 150
W2V_MODEL_NAME = 'tweet_w2v_model.json'

FILENAME_A_MERGED = 'data/semeval_train_A_merged.txt'
FILENAME_A = 'data/semeval_train_A.txt'
FILENAME_B = 'data/semeval_train_B.txt'
FILENAME_C = 'data/semeval_train_C.txt'

INDEX_A_MERGED = ['Sentiment', 'SentimentText']
INDEX_A = ['SentimentText', 'Sentiment']
INDEX_B = ['SentimentText', 'topic', 'Sentiment']
INDEX_C = ['SentimentText', 'topic', 'Sentiment']


def ingest(filename, index_names, sep='\t', encode='utf-8'):
    df = pd.read_csv(filename, sep=sep, header=None, names=index_names, encoding=encode)

    print("============================== Preview ==============================")
    print(df.head(100))
    print("============================== Summary ==============================")
    print(df.describe())
    print("=====================================================================")

    return df


def tokenize(tweet):
    tweet = tweet['SentimentText']
    try:
        tweet = tweet.lower()  # unicode(tweet.decode('utf-8').lower())
        tokens = tokenizer.tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        tokens = list(tokens)

        # Delete Stop Word
        tokens = [ w for w in tokens if not w in stop_words]

        # Stemming
        tokens = [ps.stem(w) for w in tokens]
        return tokens
    except:
        return 'NC'


def preprocess_tweet(data, n=-1):
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


class Word2VecModel:
    def __init__(self, train_data=None):
        self.train_data = train_data
        self.BASE_PATH = './W2V_MODEL/'
        self.model = None
        self.spellchecker = SpellChecker()

    def train_w2v(self, iteration=1):
        tweet_w2v = Word2Vec(size=W2V_N_DIM, min_count=10, iter=iteration)
        tweet_w2v.build_vocab([x.words for x in tqdm(self.train_data)])
        tweet_w2v.train([x.words for x in tqdm(self.train_data)], total_examples=tweet_w2v.corpus_count,
                        epochs=tweet_w2v.iter)
        self.model = tweet_w2v
        return self.model

    def save_w2v(self, filename):
        self.model.save(self.BASE_PATH + filename)

    def load_w2v(self, filename):
        model = models.Word2Vec.load(self.BASE_PATH + filename)
        self.model = model
        vocab_list = self.get_vocab_list()
        self.spellchecker.train(vocab_list)
        return model

    def get_vocab_list(self):
        corpus = list(self.model.wv.vocab.keys())
        return corpus

    def save_vocab_list(self, filename, data):
        with open(filename, 'w') as output:
            for item in data:
                output.write(item)

    def vocab2vec(self, vocabs):
        vocab_vector_dict = dict()
        for vocab in tqdm(vocabs):
            vector = self.model[vocab]
            vocab_vector_dict[vocab] = vector.tolist()
        print("Vocabulary Set converted to Vector [total: {} words]".format(len(vocab_vector_dict)))
        return vocab_vector_dict

    def save_json(self, data, filename):
        with open(self.BASE_PATH + filename, 'w') as outfile:
            json.dump(data, outfile)

    def load_json(self, filename):
        data = None
        with open(self.BASE_PATH + filename, 'r') as json_data:
            data = json.load(json_data)
        return data

    def token_spell_corrector(self, token):
        if token not in self.model.wv.vocab:
            return self.spellchecker.correct(token)
        else:
            return token

    def get_token_vector(self, token):
        token = self.token_spell_corrector(token)
        return self.model[token]

    def sentence2tokensAndpos(self, sentence):
        try:
            tweet = sentence.lower()  # unicode(tweet.decode('utf-8').lower())
            tokens = tokenizer.tokenize(tweet)
            tokens = filter(lambda t: not t.startswith('@'), tokens)
            tokens = filter(lambda t: not t.startswith('#'), tokens)
            tokens = filter(lambda t: not t.startswith('http'), tokens)
            tokens = list(tokens)

            # Delete Stop Word
            POSs = pos_tag(tokens)
            newPOSs = []
            for i, POS in enumerate(POSs):
                if tokens[i] not in stop_words:
                    newPOSs.append(POS[1])

            tokens = [w for w in tokens if w not in stop_words]

            # Stemming
            tokens = [ps.stem(w) for w in tokens]
            spellcorrectedtokens = [self.token_spell_corrector(w) for w in tokens]
            return spellcorrectedtokens, newPOSs
        except:
            return 'NC'

    def tokens2vectors(self, tokens):
        if tokens == 'NC':
            return []
        vectors = [self.get_token_vector(w) for w in tokens]
        return vectors




def increase_dataset(filename):
    data_more = ingest(filename='./data/train_more.csv',
                       index_names=['Sentiment', 'num', 'date', 'topic', 'user', 'SentimentText'], sep=',',
                       encode='latin-1')
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
    merged_data = merged_data.drop_duplicates(subset='SentimentText', keep='last')
    print(merged_data.head(10))
    print(merged_data.describe())

    merged_data.to_csv(filename, sep='\t', encoding='utf-8', index=False, header=False)
    print("data saved")


if __name__ == '__main__':
    """
    ########################################
    # Loading txt file from task A, B, C   #
    # then merged to make vocabulary set   #
    ########################################

    dataA = ingest(filename=FILENAME_A_MERGED, index_names=INDEX_A_MERGED)
    # dataA = ingest(filename=FILENAME_A, index_names=INDEX_A)

    dataB = ingest(filename=FILENAME_B, index_names=INDEX_B)
    dataB = dataB.drop('topic', axis=1)

    dataC = ingest(filename=FILENAME_C, index_names=INDEX_C)
    dataC = dataC.drop('topic', axis=1)

    data = pd.concat([dataA])
    print("loadding done")

    data = preprocess_tweet(data)
    print(data.head(100))
    print("preprocess done")

    print(" \n\n Processing Start")

    ##############################################################
    # Devide train set and test set but now set test size to 0   #
    # Train word2vec model based on train set                    #
    ##############################################################

    x_train, x_test, y_train, y_test = train_test_split(np.array(data.tokens),
                                                        np.array(data.Sentiment),
                                                        test_size=0.0)
    # Tag training data and test data
    x_train = labelizeTweets(x_train, 'TRAIN')
    x_test = labelizeTweets(x_test, 'TEST')

    #####################################################
    # Create Word2Vec, train and save word2vec model    #
    #####################################################

    model = Word2VecModel(train_data=x_train)
    model.train_w2v(iteration=15)
    print("train done")

    model.save_w2v(filename='w2v_model_ABC_merged.json')
    print("save model done")
    
    # make vocabulary list from model
    corpus = model.get_vocab_list()

    # convert each vocabulary word to voctor using model
    vector_dict = model.vocab2vec(corpus)

    # save vocab-vector dictionary into json format
    model.save_json(vector_dict, "vocab_ABC_merged.json")
    """

    ######################################
    # show how to load word2vec model    #
    ######################################
    print("==== w2v model test ====")
    model = Word2VecModel()

    # show how to load pre-trained model json file
    model.load_w2v(filename='w2v_model_ABC_merged_28004_15_stop_stem.json')
    print("load model done")

    # convert single token into vector with spell correction
    print(model.get_token_vector('he'))
    print(model.get_token_vector('king'))

    ############################################################
    # convert entire sentence into tokens list and vector list #
    ############################################################

    # Sample sentence #1
    sentence = "I am stucked because of the snow"
    tokens, poss = model.sentence2tokensAndpos("I was, in my house")
    vectors = model.tokens2vectors(tokens)
    print("converted poss", poss)
    print("converted tokens", tokens)
    print("sentence vector", vectors)

    # Sample sentence #2
    sentence = 'dec 21st 2012 will be know not as the end of the world but the Baby Boom! #2012shit'
    tokens, poss = model.sentence2tokensAndpos(sentence)
    vectors = model.tokens2vectors(tokens)
    print("converted poss", poss)
    print("converted tokens", tokens)
    # print("sentence vector", vectors)

    # Sample sentence #3
    sentence = "@MacMiller hate my life, because i can't see you at the roskilde festival on saturday, promise me to come back again, SOON."
    tokens, poss = model.sentence2tokensAndpos(sentence)
    vectors = model.tokens2vectors(tokens)
    print("converted poss", poss)
    print("converted tokens", tokens)
    # print("sentence vector", vectors)

    # show how to load vocab-vector dictionary
    # vector size is 150 dimension now but it could be changed
    print("==== vocab vector model test ====")
    vocab_vector = model.load_json("vocab_ABC_merged_28004_15_stop_stem.json")
    vocab_list = list(vocab_vector.keys())
    print(len(vocab_list))
