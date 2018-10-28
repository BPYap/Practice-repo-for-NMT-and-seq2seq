import math
import pickle
import random

import numpy as np

#  Hyper-parameters
ALPHA = 0.05
MAX_VOCAB = 10000000
LEARNING_RATE = 0.1
EPOCH = 20
MINI_EPOCH = 5

class LogLinearLM:
    def __init__(self, input_path, n):
        self.n = n
        self.vocabs = set()
        self.sentences = []
        self.initialize_vocab(input_path)
        self.vocabs = {w: i for i, w in enumerate(self.vocabs)}
        self.vocab_size = len(self.vocabs)

        self.weight = np.random.rand(self.vocab_size, (n - 1) * self.vocab_size)
        self.bias = np.random.rand(self.vocab_size)
        self.learning_rate = LEARNING_RATE

        self.unknown_words = set()

    def initialize_vocab(self, file_path):
        self.vocabs.add('<s>')
        with open(file_path, encoding='utf-8') as f:
            for line in iter(f.readline, ''):
                self.sentences.append(line)
                tokens = line.split() + ['</s>']
                for t in tokens:
                    self.vocabs.add(t)

    def load_model(self, model_path):
        print("Loading model from", model_path)
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.vocabs = data['vocab']
        self.weight = data['weight']
        self.bias = data['bias']

    @staticmethod
    def _get_ngrams(tokens, n):
        ngrams = []
        for i in range(len(tokens)):
            start = i - n + 1
            if start < 0:
                ngram = " ".join(["<s>"] + tokens[:i + 1])
            else:
                ngram = " ".join(tokens[start:i + 1])
            ngrams.append(ngram)

        return ngrams

    @staticmethod
    def softmax(x):
        orig_shape = x.shape

        if len(x.shape) > 1:
            # Matrix
            exp_minmax = lambda x: np.exp(x - np.max(x))
            denom = lambda x: 1.0 / np.sum(x)
            x = np.apply_along_axis(exp_minmax,1,x)
            denominator = np.apply_along_axis(denom,1,x) 

            if len(denominator.shape) == 1:
                denominator = denominator.reshape((denominator.shape[0],1))

            x = x * denominator
        else:
            # Vector
            x_max = np.max(x)
            x = x - x_max
            numerator = np.exp(x)
            denominator =  1.0 / np.sum(numerator)
            x = numerator.dot(denominator)

        assert x.shape == orig_shape

        return x

    def onehot(self, word):
        temp = [0.0] * self.vocab_size
        temp[self.vocabs[word]] = 1.0
        return np.array(temp)

    def extract_feature(self, words):
        context_tokens = words.split()[:-1]
        temp = []
        for t in context_tokens[::-1]:
            temp.extend(self.onehot(t))

        for _ in range(self.n - 1 - len(context_tokens)):
            temp.extend([0.0] * self.vocab_size)
        return np.array(temp)

    def compute_probability(self, words):
        feature = self.extract_feature(words)
        result = np.array([0.0] * self.vocab_size)
        for y, val in enumerate(feature):
            if val != 0:
                result += (self.weight[:,y] * val + self.bias)

        return LogLinearLM.softmax(result)

    def perform_gd(self, words):
        p = self.compute_probability(words)
        onehot = self.onehot(words.split()[-1])

        feature = self.extract_feature(words)
        for col, val in enumerate(feature):
            if val != 0:
                gradient = val * (p - onehot)
                for row in range(len(gradient)):
                    self.weight[row][col] = self.weight[row][col] - self.learning_rate * gradient[row]

        self.bias = self.bias - self.learning_rate * (p - onehot)

    def train(self, output_path):
        # perform stochastic gradient descent
        for i in range(EPOCH):
            print("Epoch:", i)
            data = self.sentences[:]
            while data:
                sent = random.choice(data)
                data.remove(sent)
                words = random.choice(LogLinearLM._get_ngrams(sent.split(), self.n))
                print("chosen words: '{}'".format(words))
                for j in range(MINI_EPOCH):
                    self.perform_gd(words)
                    loss = - math.log(self.compute_probability(words)[self.vocabs[words.split()[-1]]])
                    print("Mini-Epoch: {} Loss: {}".format(j, loss))
            print("=" * 60)

        with open(output_path, 'wb') as f: 
            pickle.dump({"weight": model.weight, "bias": model.bias, "vocab": model.vocabs}, f, pickle.HIGHEST_PROTOCOL)

    def evaluate(self, file_path):
        log_likelihood_sum = 0
        num_words = 0
        with open(file_path, encoding='utf-8') as f:
            for line in iter(f.readline, ''):
                tokens = line.split() + ['</s>']
                num_words += len(tokens)
                ngrams = self._get_ngrams(tokens, self.n)
                sentence_likelihood = 0
                for ngram in ngrams:
                    has_unknown = False
                    for word in ngram.split():
                        if word not in self.vocabs:
                            self.unknown_words.add(word)
                            has_unknown = True

                    ngram_likelihood = ALPHA / MAX_VOCAB
                    if not has_unknown:
                        word = ngram.split()[-1]
                        ngram_likelihood += (1 - ALPHA) * self.compute_probability(ngram)[self.vocabs[word]]      

                    sentence_likelihood += math.log(ngram_likelihood)

                log_likelihood_sum += sentence_likelihood

        average_log_likelihood = log_likelihood_sum / num_words
        print("log likelihood of this test set: {}, average: {}".format(log_likelihood_sum, average_log_likelihood))
        print("perplexity: {}".format(math.exp(-average_log_likelihood)))
        print("Number of unknown words: {}".format(len(self.unknown_words)))


if __name__ == '__main__':
    model = LogLinearLM("dataset\wiki-en-train.word", 2)
    # model.train("model\log-linear.pkl")    
    model.load_model("..\model\log-linear.pkl")
    model.evaluate("..\dataset\wiki-en-test.word")
