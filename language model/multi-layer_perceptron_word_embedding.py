import math
import random

import dynet as dy
import numpy as np

#  Hyper-parameters
ALPHA = 0.1
MAX_VOCAB = 10000000
EMBEDDING_DIMENSION = 16
HIDDEN_SIZE = 8
EPOCH = 30


class MLP_LM:
    def __init__(self, input_path, n):
        self.n = n
        self.vocabs = {}
        self.sentences = []
        self.initialize_vocab(input_path)
        self.vocab_size = len(self.vocabs)

        self.model = dy.Model()  # must not be garbage-collected/out of scope
        self.trainer = dy.SimpleSGDTrainer(self.model)
        self.embedding = self.model.add_lookup_parameters((self.vocab_size, EMBEDDING_DIMENSION))
        self.w1 = self.model.add_parameters((HIDDEN_SIZE, (self.n - 1) * EMBEDDING_DIMENSION))
        self.b1 = self.model.add_parameters(HIDDEN_SIZE)
        self.w2 = self.model.add_parameters((self.vocab_size, HIDDEN_SIZE))
        self.b2 = self.model.add_parameters(self.vocab_size)

        self.unknown_words = set()

    def initialize_vocab(self, file_path):
        self.vocabs['<s>'] = 0
        idx = 1
        with open(file_path, encoding='utf-8') as f:
            for line in iter(f.readline, ''):
                self.sentences.append(line)
                tokens = line.split() + ['</s>']
                for t in tokens:
                    if t not in self.vocabs:
                        self.vocabs[t] = idx
                        idx += 1

    def load_model(self, model_path):
        print("Loading model...")
        self.model.populate(model_path)

    @staticmethod
    def _get_ngrams(tokens, n):
        ngrams = []
        for i in range(n - 2, len(tokens)):
            start = i - n + 1
            if start < 0:
                ngram = " ".join(["<s>"] + tokens[:i + 1])
            else:
                ngram = " ".join(tokens[start:i + 1])
            ngrams.append(ngram)

        return ngrams

    def compute_score(self, context):
        dy.renew_cg()
        context = context.split() 
    
        if len(context) != self.n - 1:
            raise AssertionError("Expecting context of size {}, not {}".format(self.n - 1, len(context)))
        
        word = context.pop(0)
        if word not in self.vocabs:
            m = dy.zeros(EMBEDDING_DIMENSION)
        else:
            m = dy.lookup(self.embedding, self.vocabs[word])
        for word in context:
            if word not in self.vocabs:
                n = dy.zeros(EMBEDDING_DIMENSION)
            else:
                n = dy.lookup(self.embedding, self.vocabs[word])
            m = dy.concatenate([m, n])

        h = dy.rectify(self.w1 * m + self.b1)
        s = self.w2 * h + self.b2

        return s

    def train(self, output_path):
        # perform stochastic gradient descent
        for i in range(EPOCH):
            data = self.sentences[:]
            random.shuffle(data)
            total_loss = 0
            while data:
                sent = data.pop()
                ngram = random.choice(MLP_LM._get_ngrams(sent.split(), self.n))
                tokens = ngram.split()
                idx = self.vocabs[tokens[-1]]
                score = self.compute_score(" ".join(tokens[:-1]))
                loss = dy.pickneglogsoftmax(score, idx)
                total_loss += loss.value()

                loss.backward()
                self.trainer.update()

            print("Epoch: {}  Loss: {}".format(i, total_loss))
            print("=" * 60)

        self.model.save(output_path)
    
    def extract_unknown_words(self, sentence):
        tokens = sentence.split()
        for t in tokens:
            if t not in self.vocabs:
                self.unknown_words.add(t)

    def evaluate(self, file_path):
        log_likelihood_sum = 0
        num_words = 0
        with open(file_path, encoding='utf-8') as f:
            for line in iter(f.readline, ''):
                self.extract_unknown_words(line)
                tokens = line.split() + ['</s>']
                num_words += len(tokens)

                ngrams = self._get_ngrams(tokens, self.n)
                sentence_likelihood = 0
                for ngram in ngrams:
                    ngram_likelihood = ALPHA / MAX_VOCAB
                    ngram_tokens = ngram.split()
                    word = ngram_tokens[-1]
                    if word in self.vocabs:
                        s = self.compute_score(" ".join(ngram_tokens[:-1]))
                        probabilities = dy.softmax(s)
                        ngram_likelihood += (1 - ALPHA) * probabilities[self.vocabs[word]].value()     

                    sentence_likelihood += math.log(ngram_likelihood)

                log_likelihood_sum += sentence_likelihood

        average_log_likelihood = log_likelihood_sum / num_words
        print("log likelihood of this test set: {}, average: {}".format(log_likelihood_sum, average_log_likelihood))
        print("perplexity: {}".format(math.exp(-average_log_likelihood)))
        print("Number of unknown words: {}".format(len(self.unknown_words)))

    def predict(self, context):
        s = self.compute_score(context)
        probabilities = dy.softmax(s)
        max_probability = 0
        most_likely_index = 0
        for i, p in enumerate(probabilities):
            p = p.npvalue()[0]
            if p > max_probability:
                max_probability = p
                most_likely_index = i

        predicted = list(self.vocabs.keys())[most_likely_index]
        print("context: '{}'".format(context))
        print("predicted word: '{}'  confidence: {} %".format(predicted, max_probability))

if __name__ == '__main__':
    model = MLP_LM("..\dataset\wiki-en-train.word", 3)
    model.train("..\model\mlp.model")    
    model.load_model("..\model\mlp.model")
    model.evaluate("..\dataset\wiki-en-test.word")
    model.predict("Natural language")
