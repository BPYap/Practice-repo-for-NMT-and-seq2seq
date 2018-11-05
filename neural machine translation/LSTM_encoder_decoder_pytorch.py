import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu

device = torch.device('cpu')
device = torch.device('cuda')

#  Hyper-parameters
EMBEDDING_DIMENSION = 64
HIDDEN_SIZE = 16
EPOCH = 20
LEARNING_RATE = 0.1

class LSTM_Model(nn.Module):
    def __init__(self, sentences, vocab_separator=" "):
        super().__init__()

        self.vocabs = {}
        self.sentences = sentences
        self.token_separator = vocab_separator
        self.initialize_vocab(self.sentences, vocab_separator)
        self.vocab_size = len(self.vocabs)

        self.embedding = nn.Embedding(self.vocab_size, EMBEDDING_DIMENSION)
        self.lstm = nn.LSTM(EMBEDDING_DIMENSION, HIDDEN_SIZE)
        self.linear = nn.Linear(HIDDEN_SIZE, self.vocab_size)

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

    def get_embedding_lookups(self, sentence):
        if self.token_separator == "":
            lookups = [self.vocabs[t] for t in sentence]
        else:
            lookups = [self.vocabs[t] for t in sentence.split(self.token_separator)]
        lookups = torch.tensor(lookups, dtype=torch.long, device=device)

        return lookups

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.to(device)

    def init_hidden(self, initial_hidden_state=None):
        if initial_hidden_state is None:
            hidden_state = torch.zeros(1, 1, HIDDEN_SIZE, device=device)
        else:
            hidden_state = initial_hidden_state.view(1, 1, -1)
        return (hidden_state, torch.zeros(1, 1, HIDDEN_SIZE, device=device))

    def forward(self, lookups, hidden_state=None):
        embeddings = self.embedding(lookups)
        hidden_and_cell = self.init_hidden(hidden_state)
        hidden_states, hidden_and_cell = self.lstm(embeddings.view(len(lookups), 1, -1), hidden_and_cell)

        predictions = self.linear(hidden_states.view(len(lookups), -1))
        return F.log_softmax(predictions, dim=1)


class LSTM_Encoder(LSTM_Model):
    def __init__(self, sentences):
        super().__init__(sentences)

        self.unknown_words = set()

    def get_embedding_lookups(self, sentence):
        if self.token_separator == "":
            lookups = [self.vocabs[t] for t in sentence]
        else:
            lookups = [self.vocabs[t] for t in sentence.split(self.token_separator)]
        lookups = torch.tensor(lookups, dtype=torch.long, device=device)

        return lookups

    def train(self, output_path):
        self.to(device)
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.parameters(), lr=LEARNING_RATE)

        for i in range(EPOCH):
            data = self.sentences[:]
            random.shuffle(data)
            total_loss = 0
            while data:
                self.zero_grad()
                sent = data.pop()
                lookups = self.get_embedding_lookups(sent)
                predictions = self.forward(lookups)

                end_column = torch.tensor(self.vocabs['</s>'], dtype=torch.long, device=device).view(1)
                targets = torch.cat([lookups[1:], end_column])
                loss = loss_function(predictions, targets)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            print("Epoch: {}  Loss: {}".format(i, total_loss))
            print("=" * 60)
        print("saving to", output_path)
        torch.save(self.state_dict(), output_path)

    def evaluate(self, sentences):
        log_likelihood_sum = 0
        num_words = 0
        with torch.no_grad():
            for sent in sentences:
                if not all([t in self.vocabs for t in sent.split()]):
                    continue  # TODO: ignore unknown words for now
                predictions = self.forward(self.get_embedding_lookups(sent))
                tokens = sent.split()[1:] + ["</s>"]
                num_words += len(tokens) + 1
                sentence_likelihood = 0
                for i, prediction in enumerate(predictions):
                    sentence_likelihood += prediction[i].item()

                log_likelihood_sum += sentence_likelihood

        average_log_likelihood = log_likelihood_sum / num_words
        print("log likelihood of this test set: {}, average: {}".format(log_likelihood_sum, average_log_likelihood))
        print("perplexity: {}".format(math.exp(-average_log_likelihood)))
        print("Number of unknown words: {}".format(len(self.unknown_words)))

    def encode(self, sentence):
        with torch.no_grad():
            lookups = self.get_embedding_lookups(sentence)
            embeddings = self.embedding(lookups)
            hidden_and_cell = self.init_hidden()
            hidden_states, hidden_and_cell = self.lstm(embeddings.view(len(lookups), 1, -1), hidden_and_cell)
            return hidden_and_cell[-1].view(1, HIDDEN_SIZE).view(HIDDEN_SIZE)

    def predict(self, context):
        with torch.no_grad():
            prediction = self.forward(self.get_embedding_lookups(context))[-1]
            idx = torch.argmin(prediction).item()
            max_probability = F.softmax(prediction, 0)[idx].item()

        predicted = list(self.vocabs.keys())[idx]
        print("context: '{}'".format(context))
        print("predicted word: '{}'  confidence: {} %".format(predicted, max_probability))


class LSTM_Decoder(LSTM_Model):
    def __init__(self, sentences, encoder):
        super().__init__(sentences, vocab_separator="")
        self.encoder = encoder

    def train(self, output_path):
        self.to(device)
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.parameters(), lr=LEARNING_RATE)

        for i in range(EPOCH):
            data = self.sentences[:]
            random.shuffle(data)
            total_loss = 0
            while data:
                self.zero_grad()
                sent = data.pop()
                lookups = self.get_embedding_lookups("<s> " + sent)
                source = self.encoder.sentences[self.sentences.index(sent)]
                predictions = self.forward(lookups, self.encoder.encode(source))

                end_column = torch.tensor(self.vocabs['</s>'], dtype=torch.long, device=device).view(1)
                targets = torch.cat([lookups[1:], end_column])
                loss = loss_function(predictions, targets)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            print("Epoch: {}  Loss: {}".format(i, total_loss))
            print("=" * 60)

        torch.save(self.state_dict(), output_path)

    def evaluate(self, source_sentences, target_sentences):
        cumulative_bleu = 0
        for i in range(len(source_sentences)):
            translated = self.translate(source_sentences[i])
            cumulative_bleu += sentence_bleu(target_sentences[i], translated)

        print("Average BLEU score: {}".format(cumulative_bleu / len(source_sentences)))

    def translate(self, source):
        vocabs = list(self.vocabs.keys())
        translated = ""
        sent_probability = 1

        curr_length = 0
        previous_word = "<s>"
        hidden_and_cell = self.init_hidden(self.encoder.encode(source))
        with torch.no_grad():
            while True:
                embedding = self.embedding(self.get_embedding_lookups(previous_word))
                hidden_state, hidden_and_cell = self.lstm(embedding.view(1, 1, -1), hidden_and_cell)

                prediction = self.linear(hidden_state.view(1, -1))
                most_likely_index = torch.argmin(F.log_softmax(prediction, dim=1))
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
        self.encoder.load_model(model_dir + "lstm_encoder.pt")
        print("Loading decoder model...")
        self.decoder.load_model(model_dir + "lstm_decoder.pt")

    def train(self, model_dir):
        print("Training LSTM Encoder...")
        self.encoder.train(model_dir + "lstm_encoder.pt")
        print("Training LSTM Decoder...")
        self.decoder.train(model_dir + "lstm_decoder.pt")

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
        self.encoder.predict("Natural language")

        print("Evaluating Decoder...")
        self.decoder.evaluate(test_source_sentences, test_target_sentences)
        self.decoder.translate("Invisible car created by German engineer")
        self.decoder.translate("Guess what our dogs name is?")


if __name__ == '__main__':
    model = NeuralTranslator("..\dataset\microtopia-train.en-zh")
    model.train("..\model\\")    
    model.load_model("..\model\\")
    model.evaluate("..\dataset\microtopia-test.en-zh")
