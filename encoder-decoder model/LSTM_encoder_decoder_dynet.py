import math
import random

import dynet_config
dynet_config.set_gpu()
import dynet as dy
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

#  Hyper-parameters
ALPHA = 0.1
MAX_VOCAB = 10000000
EMBEDDING_DIMENSION = 16
HIDDEN_SIZE = 8
EPOCH = 30

class LSTM_Model:
    def __init__(self, sentences, vocab_separator=" "):
        self.vocabs = {}
        self.sentences = sentences
        self.initialize_vocab(self.sentences, vocab_separator)
        self.vocab_size = len(self.vocabs)

        self.params = dy.ParameterCollection()
        self.trainer = dy.SimpleSGDTrainer(self.params)

        self.embedding = self.params.add_lookup_parameters((self.vocab_size, EMBEDDING_DIMENSION))

        # Parameters for input gate
        self.w_xi = self.params.add_parameters((HIDDEN_SIZE, EMBEDDING_DIMENSION))
        self.w_hi = self.params.add_parameters((HIDDEN_SIZE, HIDDEN_SIZE))
        self.b_i = self.params.add_parameters(HIDDEN_SIZE)

        # Parameters for forget gate
        self.w_xf = self.params.add_parameters((HIDDEN_SIZE, EMBEDDING_DIMENSION))
        self.w_hf = self.params.add_parameters((HIDDEN_SIZE, HIDDEN_SIZE))
        self.b_f = self.params.add_parameters(HIDDEN_SIZE)

        # Parameters for output gate
        self.w_xo = self.params.add_parameters((HIDDEN_SIZE, EMBEDDING_DIMENSION))
        self.w_ho = self.params.add_parameters((HIDDEN_SIZE, HIDDEN_SIZE))
        self.b_o = self.params.add_parameters(HIDDEN_SIZE)

        # Parameters for update
        self.w_xu = self.params.add_parameters((HIDDEN_SIZE, EMBEDDING_DIMENSION))
        self.w_hu = self.params.add_parameters((HIDDEN_SIZE, HIDDEN_SIZE))
        self.b_u = self.params.add_parameters(HIDDEN_SIZE)

        # Parameters for memory cell
        self.w_xc = self.params.add_parameters((HIDDEN_SIZE, EMBEDDING_DIMENSION))
        self.w_hc = self.params.add_parameters((HIDDEN_SIZE, HIDDEN_SIZE))
        self.b_c = self.params.add_parameters(HIDDEN_SIZE)

        # Parameters for score vector computation
        self.w_s = self.params.add_parameters((self.vocab_size, HIDDEN_SIZE))
        self.b_s = self.params.add_parameters(self.vocab_size)

    def initialize_vocab(self, sentences, vocab_separator):
        self.vocabs['<s>'] = 0
        idx = 1
        for sent in sentences:
            if vocab_separator == "":
                tokens = [t for t in sent] + ['</s>']
            else:
                tokens = sent.split(vocab_separator) + ['</s>']
            for t in tokens:
                if t not in self.vocabs:
                    self.vocabs[t] = idx
                    idx += 1

    def load_model(self, model_path):
        self.params.populate(model_path)

    def lstm_step(self, word, h_prev, c_prev):
        if h_prev is None:
            h_prev = dy.zeros(HIDDEN_SIZE)
        if c_prev is None:
            c_prev = dy.zeros(HIDDEN_SIZE)

        if word not in self.vocabs:
            x = dy.zeros(EMBEDDING_DIMENSION) 
        else:
            x = dy.lookup(self.embedding, self.vocabs[word])

        u = dy.tanh(self.w_xu * x + self.w_hu * h_prev + self.b_u)
        i = dy.logistic(self.w_xi * x + self.w_hi * h_prev + self.b_i)
        f = dy.logistic(self.w_xf * x + self.w_hf * h_prev + self.b_f)
        c = dy.cmult(i, u) + dy.cmult(f, c_prev)

        o = dy.logistic(self.w_xo * x + self.w_ho * h_prev + self.b_o)
        h = dy.cmult(o, dy.tanh(c))

        return h, c

    def compute_word_loss(self, next_word, hidden_state):
        score = self.w_s * hidden_state + self.b_s
        return dy.pickneglogsoftmax(score, self.vocabs[next_word])


class LSTM_Encoder(LSTM_Model):
    def __init__(self, sentences):
        super().__init__(sentences)

        self.unknown_words = set()

    def compute_loss(self, sentence):
        dy.renew_cg()
        tokens = sentence.split() + ["</s>"]
        sentence_loss = None
        hidden_state = None
        cell_state = None
        while len(tokens) > 1:
            hidden_state, cell_state = self.lstm_step(tokens.pop(0), hidden_state, cell_state)
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
                sentence_loss = self.compute_loss(sent)
                total_loss += sentence_loss.value()
                sentence_loss.backward()
                self.trainer.update()

            print("Epoch: {}  Loss: {}".format(i, total_loss))
            print("=" * 60)

        self.params.save(output_path)

    def evaluate(self, sentences):
        log_likelihood_sum = 0
        num_words = 0
        for sent in sentences:
            tokens = ["<s>"] + sent.split() + ["</s>"]
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
                    cell_state = None
                    for j in range(i):
                        hidden_state, cell_state = self.lstm_step(tokens[j], hidden_state, cell_state)

                    probabilities = dy.softmax(self.w_s * hidden_state + self.b_s)
                    word_likelihood += (1 - ALPHA) * probabilities[self.vocabs[predicting_word]].value()

                sentence_likelihood += math.log(word_likelihood)

            log_likelihood_sum += sentence_likelihood

        average_log_likelihood = log_likelihood_sum / num_words
        print("log likelihood of this test set: {}, average: {}".format(log_likelihood_sum, average_log_likelihood))
        print("perplexity: {}".format(math.exp(-average_log_likelihood)))
        print("Number of unknown words: {}".format(len(self.unknown_words)))

    def encode(self, sentence):
        tokens = ["<s>"] + sentence.split() + ["</s>"]
        dy.renew_cg()
        hidden_state = None
        cell_state = None
        while tokens:
            hidden_state, cell_state = self.lstm_step(tokens.pop(0), hidden_state, cell_state)
        return hidden_state

    def predict(self, context):
        hidden_state = self.encode(context)
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


class LSTM_Decoder(LSTM_Model):
    def __init__(self, sentences, encoder):
        super().__init__(sentences, vocab_separator="")
        self.encoder = encoder

    def compute_loss(self, source, target):
        dy.renew_cg()
        tokens = ["<s>"] + [t for t in target] + ["</s>"]
        sentence_loss = None
        hidden_state = self.encoder.encode(source)
        cell_state = None
        while len(tokens) > 1:
            hidden_state, cell_state = self.lstm_step(tokens.pop(0), hidden_state, cell_state)
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
                source = self.encoder.sentences[self.sentences.index(sent)]
                sentence_loss = self.compute_loss(source, sent)
                total_loss += sentence_loss.value()
                sentence_loss.backward()
                self.trainer.update()

            print("Epoch: {}  Loss: {}".format(i, total_loss))
            print("=" * 60)

        self.params.save(output_path)

    def evaluate(self, source_sentences, target_sentences):
        cumulative_bleu = 0
        for i in range(len(source_sentences)):
            translated = self.translate(source_sentences[i])
            cumulative_bleu += sentence_bleu(target_sentences[i], translated)

        print("Average BLEU score: {}".format(cumulative_bleu / len(source_sentences)))

    def translate(self, source):
        dy.renew_cg()

        vocabs = list(self.vocabs.keys())
        translated = ""
        sent_probability = 1

        hidden_state = self.encoder.encode(source)
        cell_state = None
        
        curr_length = 0
        previous_word = "<s>"
        while True:
            hidden_state, cell_state = self.lstm_step(previous_word, hidden_state, cell_state)

            s = self.w_s * hidden_state + self.b_s
            probabilities = dy.softmax(s)
            max_probability = 0
            most_likely_index = 0
            for i, p in enumerate(probabilities):
                p = p.npvalue()[0]
                if p > max_probability:
                    max_probability = p
                    most_likely_index = i

            translated_token = vocabs[most_likely_index]
            previous_word = translated_token
            if translated_token == '</s>' or curr_length > 20:
                break
            else:
                translated += translated_token
                sent_probability *= max_probability
                curr_length += 1

        print("source sentence: '{}'".format(source))
        print("translated sentence: '{}'  confidence: {} %".format(translated, sent_probability))

        return translated


class NeuralTranslator:
    def __init__(self, input_path):
        self.source_sentences = []
        self.target_sentences = []
        self.parse_parallel_corpus(input_path)

        self.encoder = LSTM_Encoder(self.source_sentences)
        self.decoder = LSTM_Decoder(self.target_sentences, self.encoder)

    def parse_parallel_corpus(self, file_path):
        print("Parsing corpus...")
        with open(file_path, encoding='utf-8') as f:
            for line in iter(f.readline, ''):
                sentence_pair = line.split('|||')
                source = sentence_pair[0].strip()
                target = sentence_pair[1].strip()
                self.source_sentences.append(source)
                self.target_sentences.append(target)

    def load_model(self, model_dir):
        print("Loading encoder model...")
        self.encoder.load_model(model_dir + "lstm_encoder.model")
        print("Loading decoder model...")
        self.decoder.load_model(model_dir + "lstm_decoder.model")

        # print(self.decoder.vocabs)
        self.decoder.translate("Invisible car created by German engineer")
        self.decoder.translate("Guess what our dogs name is?")

    def train(self, model_dir):
        print("Training LSTM Encoder...")
        # self.encoder.train(model_dir + "lstm_encoder.model")
        print("Training LSTM Decoder...")
        # self.decoder.train(model_dir + "lstm_decoder.model")

    def evaluate(self, file_path):
        test_source_sentences = []
        test_target_sentences = []
        with open(file_path, encoding='utf-8') as f:
            for line in iter(f.readline, ''):
                sentence_pair = line.split('|||')
                source = sentence_pair[0].strip()
                target = sentence_pair[1].strip()
                test_source_sentences.append(source)
                test_target_sentences.append(target)

        print("Evaluating Encoder...")
        self.encoder.evaluate(test_source_sentences)

        print("Evaluating Decoder...")
        # self.decoder.evaluate(test_source_sentences, test_target_sentences)


if __name__ == '__main__':
    model = NeuralTranslator("..\dataset\microtopia-train.en-zh")
    model.train("..\model\\")    
    model.load_model("..\model\\")
    model.evaluate("..\dataset\microtopia-test.en-zh")
