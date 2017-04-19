from .functional import *


class Parameters:
    def __init__(self, max_word_len, max_seq_len, word_vocab_size):

        self.max_word_len = int(max_word_len)
        self.max_seq_len = int(
            max_seq_len) + 1  # go or eos token

        self.word_vocab_size = int(word_vocab_size)

        self.word_embed_size = 300
