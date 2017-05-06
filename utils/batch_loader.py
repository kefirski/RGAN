import collections
import os
import numpy as np
from six.moves import cPickle
from .functional import *
import torch as t
from torch.autograd import Variable


class BatchLoader:
    def __init__(self, path=''):

        path += 'data/'

        self.preprocessing_path = path + 'preprocessings/'
        if not os.path.exists(self.preprocessing_path):
            print(self.preprocessing_path)
            os.makedirs(self.preprocessing_path)

        self.data_files = [path + 'train.txt',
                           path + 'test.txt']

        self.idx_file = self.preprocessing_path + 'words_vocab.pkl'

        self.tensor_files = [self.preprocessing_path + 'train_word_tensor.npy',
                             self.preprocessing_path + 'valid_word_tensor.npy']

        self.pad_token = '_'
        self.go_token = '>'
        self.end_token = '|'

        idx_exists = os.path.exists(self.idx_file)
        tensors_exists = all([os.path.exists(target) for target in self.tensor_files])

        if idx_exists and tensors_exists:
            self.load_preprocessed(self.data_files,
                                   self.idx_file,
                                   self.tensor_files)
            print('preprocessed data was found and loaded')
        else:
            self.preprocess(self.data_files,
                            self.idx_file,
                            self.tensor_files)
            print('data have preprocessed')

        self.word_embedding_index = 0

    def build_vocab(self, sentences):
        """
        build_vocab(self, sentences) -> vocab_size, idx_to_word, word_to_idx
            vocab_size - number of unique words in corpus
            idx_to_word - array of shape [vocab_size] containing ordered list of unique words
            word_to_idx - dictionary of shape [vocab_size]
                such that idx_to_char[idx_to_word[some_word]] = some_word
                where some_word is such that idx_to_word contains it
        """

        word_counts = collections.Counter(sentences)

        idx_to_word = [x[0] for x in word_counts.most_common()]
        idx_to_word = list(sorted(idx_to_word)) + [self.pad_token, self.go_token, self.end_token]

        word_to_idx = {x: i for i, x in enumerate(idx_to_word)}

        words_vocab_size = len(idx_to_word)

        return words_vocab_size, idx_to_word, word_to_idx

    def preprocess(self, data_files, idx_file, tensor_paths):

        data = [open(file, "r").read() for file in data_files]

        data_words = [[line.split() for line in target.split('\n')] for target in data]
        self.max_seq_len = np.amax([len(line) for target in data_words for line in target])
        self.num_lines = [len(target) for target in data_words]

        '''split whole data and build vocabulary from it'''
        merged_data_words = (data[0] + '\n' + data[1]).split()
        self.vocab_size, self.idx_to_word, self.word_to_idx = self.build_vocab(merged_data_words)
        self.max_word_len = np.amax([len(word) for word in self.idx_to_word])

        with open(idx_file, 'wb') as f:
            cPickle.dump(self.idx_to_word, f)

        self.data_tensor = np.array([[list(map(self.word_to_idx.get, line)) for line in target]
                                     for target in data_words])
        for target, path in enumerate(tensor_paths):
            np.save(path, self.data_tensor[target])

            '''uses to pick up data pairs for embedding learning'''
        self.embed_pairs = np.array([pair for line in self.data_tensor[0] for pair in BatchLoader.bag_window(line, 3)])

    def load_preprocessed(self, data_files, idx_file, tensor_paths):

        data = [open(file, "r").read() for file in data_files]
        data_words = [[line.split() for line in target.split('\n')] for target in data]

        self.max_seq_len = np.amax([len(line) for target in data_words for line in target])
        self.num_lines = [len(target) for target in data_words]

        self.idx_to_word = cPickle.load(open(idx_file, "rb"))
        self.word_to_idx = dict(zip(self.idx_to_word, range(len(self.idx_to_word))))
        self.vocab_size = len(self.idx_to_word)

        self.max_word_len = np.amax([len(word) for word in self.idx_to_word])

        self.data_tensor = np.array([np.load(target) for target in tensor_paths])

        self.embed_pairs = np.array([pair for line in self.data_tensor[0] for pair in BatchLoader.bag_window(line, 3)])

    def true_data(self, batch_size, target: str):
        """
        :param batch_size: number of selected data elements 
        :param target: whether to use train or valid data source
        :return: target tensor
        """

        target = 0 if target == 'train' else 1

        indexes = np.array(np.random.randint(self.num_lines[target], size=batch_size))

        target = [self.data_tensor[target][index] for index in indexes]
        target = [line + [self.word_to_idx[self.end_token]] for line in target]

        max_input_seq_len = np.amax([len(line) for line in target])

        for i, line in enumerate(target):
            line_len = len(line)
            to_add = max_input_seq_len - line_len
            target[i] = line + [self.word_to_idx[self.pad_token]] * to_add

        return np.array(target)

    def next_embedding_seq(self, seq_len):
        """
        :returns: seq_len pairs from embed_pairs
        """

        embed_len = len(self.embed_pairs)

        seq = np.array([self.embed_pairs[i % embed_len]
                        for i in np.arange(self.word_embedding_index, self.word_embedding_index + seq_len)])

        self.word_embedding_index = (self.word_embedding_index + seq_len) % embed_len

        return seq[:, 0], seq[:, 1]

    @staticmethod
    def bag_window(seq, window=1):
        """
        :return: input and output for word embeddings learning,
                 where input = [b, b, c, c, d, d, e, e]
                 and output  = [a, c, b, d, c, e, d, g]
                 for seq [a, b, c, d, e, g]
                 with window equal to 1
        """

        assert window >= 1 and isinstance(window, int)

        '''input, target'''
        result = [[], []]
        seq_len = len(seq)

        for i, element in enumerate(seq):
            for j in range(i + 1, i + window + 1):
                if j < seq_len:
                    result[0] += [element]
                    result[1] += [seq[j]]
            for j in range(i - 1, i - window - 1, -1):
                if j >= 0:
                    result[0] += [element]
                    result[1] += [seq[j]]

        return np.array(result).transpose()

    def go_input(self, batch_size, use_cuda):
        go_input = np.array([self.word_to_idx[self.go_token]] * batch_size)
        go_input = Variable(t.from_numpy(go_input)).long()
        if use_cuda:
            go_input = go_input.cuda()
        return go_input

    def encode_word(self, word):

        idx = self.word_to_idx[word]
        result = np.zeros(self.vocab_size)
        result[idx] = 1
        return result

    def decode_word(self, distribution):

        ix = np.random.choice(range(self.vocab_size), p=distribution.ravel())
        x = np.zeros((self.vocab_size, 1))
        x[ix] = 1
        return self.idx_to_word[np.argmax(x)]
        # return self.idx_to_word[np.argmax(distribution)]

    def input_data(self, batch_size, use_cuda, params):

        true_data = self.true_data(batch_size, 'train')
        true_data = Variable(t.from_numpy(true_data)).long()

        z = Variable(t.rand([batch_size, params.latent_variable_size]))

        if use_cuda:
            true_data = true_data.cuda()
            z = z.cuda()

        return z, true_data

    @staticmethod
    def sample_z(batch_size, use_cuda, params):

        z = Variable(t.rand([batch_size, params.latent_variable_size]))
        if use_cuda:
            z = z.cuda()

        return z
