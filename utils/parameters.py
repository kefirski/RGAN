from .functional import *


class Parameters:
    def __init__(self, max_word_len, max_seq_len, vocab_size):

        self.max_word_len = int(max_word_len)
        self.max_seq_len = int(max_seq_len) + 1  # + eos token

        self.vocab_size = int(vocab_size)
        self.word_embed_size = 300

        self.latent_variable_size = 300

        ''' 
        generator takes word embedding size shaped input 
            and emit probability disctribution over various words in vocabulary
        discriminator takes word embedding size shaped input 
            and produce single number to discriminate generated data from original one
        '''
        self.gen_size = [self.latent_variable_size + self.word_embed_size, 800]
        self.dis_size = [self.word_embed_size, 100, 75, 50]