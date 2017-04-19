from .functional import *


class Parameters:
    def __init__(self, max_word_len, max_seq_len, vocab_size):

        self.max_word_len = int(max_word_len)
        self.max_seq_len = int(
            max_seq_len) + 1  # go or eos token

        self.vocab_size = int(vocab_size)

        self.word_embed_size = 270
