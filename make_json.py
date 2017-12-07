import json
from preprocessing import load_w2v, get_vocab_list, vocab2vec, save_json
import gensim
import numpy as np

from gensim.models.word2vec import Word2Vec


BASE_PATH = './W2V_MODEL/'

w2v_model = load_w2v(filename='w2v_model_ABC_merged_28004_15_stop_stem.json')
corpus = get_vocab_list(w2v_model)
vector_dict = vocab2vec(w2v_model, corpus)
save_json(vector_dict, "vocab_vector1.json")
