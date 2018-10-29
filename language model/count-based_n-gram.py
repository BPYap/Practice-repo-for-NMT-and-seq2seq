import math
from collections import defaultdict

#  Hyper-parameters
ALPHA = 0.05
MAX_VOCAB = 10000000

class NgramLM:
    def __init__(self, n):
        self.n = n
        self.context_count = defaultdict(int)
        self.vocab_count = defaultdict(int)
        if n == 1:
            #  ngram_count points to vocab_count if n == 1
            self.ngram_count = self.vocab_count
        else:
            self.ngram_count = defaultdict(int)
        self.word_count = 0
        self.param = {}
        self.unknown_words = set()

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

    def train(self, file_path):
        # parse the whole training corpus
        with open(file_path, encoding='utf-8') as f:
            for line in iter(f.readline, ''):
                tokens = line.split() + ['</s>']
                self.vocab_count['<s>'] += 1
                for t in tokens:
                    self.vocab_count[t] += 1
                    self.word_count += 1

                if self.n != 1:
                    ngrams = self._get_ngrams(tokens, self.n)
                    for ngram in ngrams:
                        self.ngram_count[ngram] += 1
                        context = " ".join(ngram.split()[:-1])
                        self.context_count[context] += 1

        # calculate the maximum likelihood estimation for each ngram
        for ngram in self.ngram_count.keys():
            if self.n == 1:
                self.param[ngram] = self.vocab_count[ngram] / self.word_count
            else:
                context = " ".join(ngram.split()[:-1])
                self.param[ngram] = self.ngram_count[ngram] / self.context_count[context]

        # print(self.param)

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
                    # smoothing/ interpolation for unknown words
                    if self.n == 1:
                        ngram_likelihood = ALPHA / MAX_VOCAB
                    else:
                        has_unknown = False

                        for w in ngram.split():
                            if w not in self.vocab_count:
                                self.unknown_words.add(w)
                                has_unknown = True

                        if has_unknown:
                            ngram_likelihood = ALPHA / MAX_VOCAB
                        else:
                            word = ngram.split()[-1]
                            ngram_likelihood = ALPHA * self.vocab_count[word] / self.word_count

                    if ngram in self.param:
                        ngram_likelihood += (1 - ALPHA) * self.param[ngram]                 

                    sentence_likelihood += math.log(ngram_likelihood)

                log_likelihood_sum += sentence_likelihood

        average_log_likelihood = log_likelihood_sum / num_words
        print("log likelihood of this test set: {}, average: {}".format(log_likelihood_sum, average_log_likelihood))
        print("perplexity: {}".format(math.exp(-average_log_likelihood)))
        print("Number of unknown words: {}".format(len(self.unknown_words)))


if __name__ == '__main__':
    model = NgramLM(3)
    model.train("..\dataset\wiki-en-train.word")
    model.evaluate("..\dataset\wiki-en-test.word")
