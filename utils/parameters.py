from .functional import *


class Parameters:
    def __init__(self, max_word_len, max_seq_len, vocab_size):

        self.max_word_len = int(max_word_len)
        self.max_seq_len = int(
            max_seq_len) + 1  # go or eos token

        self.vocab_size = int(vocab_size)
        self.word_embed_size = 270

        self.latent_variable_size = 500

        self.decoder_size = [450, 425, 400]
        self.decoder_num_layers = len(self.decoder_size)
