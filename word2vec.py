import gensim, logging
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            print(fname)
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

if __name__ == '__main__':
    sentences = MySentences('./datas')  # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences)
    model.save('mymodel.json')

    # model = gensim.models.Word2Vec.load('mymodel.json')
    print(model['the'])
    print(model.most_similar('Any'))
    print(model.similarity('king', 'man'))

