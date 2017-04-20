from .functional import *


class Parameters:
    def __init__(self, max_word_len, max_seq_len, vocab_size):

        self.max_word_len = int(max_word_len)
        self.max_seq_len = int(
            max_seq_len) + 1  # go or eos token

        self.vocab_size = int(vocab_size)
        self.word_embed_size = 250

        self.latent_variable_size = 420

        ''' 
        generator takes latent_variable_size shaped input 
            and emit probability disctribution over various words in vocabulary
        discriminator takes word embedding size shape input 
            and produce single number to discriminate generated data from original one
        '''
        self.gen_size = [self.latent_variable_size, 400, 380, 350]
        self.dis_size = [self.word_embed_size, 200, 180, 140, 80]