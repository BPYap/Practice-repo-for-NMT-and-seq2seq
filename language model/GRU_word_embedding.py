import math
import random

import dynet_config
dynet_config.set_gpu()
import dynet as dy
import numpy as np

#  Hyper-parameters
ALPHA = 0.1
MAX_VOCAB = 10000000
EMBEDDING_DIMENSION = 16
HIDDEN_SIZE = 8
EPOCH = 30


class GRU_LM:
    def __init__(self, input_path):
        self.vocabs = {}
        self.sentences = []
        self.initialize_vocab(input_path)
        self.vocab_size = len(self.vocabs)

        self.params = dy.ParameterCollection()
        self.trainer = dy.SimpleSGDTrainer(self.params)

        self.embedding = self.params.add_lookup_parameters((self.vocab_size, EMBEDDING_DIMENSION))

        # Parameters for reset gate
        self.w_xr = self.params.add_parameters((HIDDEN_SIZE, EMBEDDING_DIMENSION))
        self.w_hr = self.params.add_parameters((HIDDEN_SIZE, HIDDEN_SIZE))
        self.b_r = self.params.add_parameters(HIDDEN_SIZE)

        # Parameters for update gate
        self.w_xz = self.params.add_parameters((HIDDEN_SIZE, EMBEDDING_DIMENSION))
        self.w_hz = self.params.add_parameters((HIDDEN_SIZE, HIDDEN_SIZE))
        self.b_z = self.params.add_parameters(HIDDEN_SIZE)

        # Parameters for candidate hidden state
        self.w_xh = self.params.add_parameters((HIDDEN_SIZE, EMBEDDING_DIMENSION))
        self.w_hh = self.params.add_parameters((HIDDEN_SIZE, HIDDEN_SIZE))
        self.b_h = self.params.add_parameters(HIDDEN_SIZE)
        
        # Parameters for score vector computation
        self.w_s = self.params.add_parameters((self.vocab_size, HIDDEN_SIZE))
        self.b_s = self.params.add_parameters(self.vocab_size)

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
        self.params.populate(model_path)

    def gru_step(self, word, h_prev):
        if h_prev is None:
            h_prev = dy.zeros(HIDDEN_SIZE)
        if word not in self.vocabs:
            x = dy.zeros(EMBEDDING_DIMENSION) 
        else:
            x = dy.lookup(self.embedding, self.vocabs[word])
        r = dy.logistic(self.w_xr * x + self.w_hr * h_prev + self.b_r)
        z = dy.logistic(self.w_xz * x + self.w_hz * h_prev + self.b_z)
        c_h = dy.tanh(self.w_xh * x + self.w_hh * dy.cmult(r, h_prev) + self.b_h)
        return dy.cmult(1 - z, h_prev) + dy.cmult(z, c_h)

    def compute_word_loss(self, next_word, hidden_state):
        score = self.w_s * hidden_state + self.b_s
        return dy.pickneglogsoftmax(score, self.vocabs[next_word])

    def compute_sentence_loss(self, sentence):
        dy.renew_cg()
        tokens = sentence.split() + ["</s>"]
        sentence_loss = None
        hidden_state = None
        while len(tokens) > 1:
            hidden_state = self.gru_step(tokens.pop(0), hidden_state)
            if sentence_loss is None:
                sentence_loss = self.compute_word_loss(tokens[0], hidden_state)
            else:
                sentence_loss += self.compute_word_loss(tokens[0], hidden_state)

        return sentence_loss

    def train(self, output_path):
        # perform stochastic gradient descent
        for i in range(EPOCH):
            data = self.sentences[:]
            random.shuffle(data)
            total_loss = 0
            while data:
                sent = data.pop()
                sentence_loss = self.compute_sentence_loss(sent)
                total_loss += sentence_loss.value()
                sentence_loss.backward()
                self.trainer.update()

            print("Epoch: {}  Loss: {}".format(i, total_loss))
            print("=" * 60)

        self.params.save(output_path)

    def evaluate(self, file_path):
        log_likelihood_sum = 0
        num_words = 0
        with open(file_path, encoding='utf-8') as f:
            for line in iter(f.readline, ''):
                tokens = ["<s>"] + line.split() + ["</s>"]
                num_words += len(tokens) - 1

                sentence_likelihood = 0
                for i in range(1, len(tokens)):
                    predicting_word = tokens[i]
                    word_likelihood = ALPHA / MAX_VOCAB
                    if predicting_word not in self.vocabs:
                        self.unknown_words.add(predicting_word)
                    else:
                        dy.renew_cg()
                        hidden_state = None
                        for j in range(i):
                            hidden_state = self.gru_step(tokens[j], hidden_state)

                        probabilities = dy.softmax(self.w_s * hidden_state + self.b_s)
                        word_likelihood += (1 - ALPHA) * probabilities[self.vocabs[predicting_word]].value()

                    sentence_likelihood += math.log(word_likelihood)

                log_likelihood_sum += sentence_likelihood

        average_log_likelihood = log_likelihood_sum / num_words
        print("log likelihood of this test set: {}, average: {}".format(log_likelihood_sum, average_log_likelihood))
        print("perplexity: {}".format(math.exp(-average_log_likelihood)))
        print("Number of unknown words: {}".format(len(self.unknown_words)))

    def predict(self, context):
        dy.renew_cg()
        tokens = context.split()
        hidden_state = None
        while len(tokens) > 0:
            hidden_state = self.gru_step(tokens.pop(0), hidden_state)
        s = self.w_s * hidden_state + self.b_s
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
    model = GRU_LM("..\dataset\wiki-en-train.word")
    model.train("..\model\gru.model")    
    model.load_model("..\model\gru.model")
    model.evaluate("..\dataset\wiki-en-test.word")
    model.predict("Natural language")
