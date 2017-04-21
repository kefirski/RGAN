import argparse
import numpy as np
import torch as t
from torch.autograd import Variable
from torch.optim import SGD
from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from torch_modules.losses.neg_loss import NEG_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='word2vec')
    parser.add_argument('--num-iterations', type=int, default=10, metavar='NI',
                        help='num iterations (default: 15000000)')
    parser.add_argument('--batch-size', type=int, default=15, metavar='BS',
                        help='batch size (default: 15)')
    parser.add_argument('--num-sample', type=int, default=8, metavar='NS',
                        help='num sample (default: 8)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='whether to use cuda (default: True)')
    args = parser.parse_args()

    batch_loader = BatchLoader('')
    params = Parameters(batch_loader.max_word_len,
                        batch_loader.max_seq_len,
                        batch_loader.vocab_size)

    neg_loss = NEG_loss(params.vocab_size, params.word_embed_size)
    if args.use_cuda:
        neg_loss = neg_loss.cuda()

    # NEG_loss is defined over two embedding matrixes with shape of [params.word_vocab_size, params.word_embed_size]
    optimizer = SGD(neg_loss.parameters(), 0.1)

    for iteration in range(args.num_iterations):

        input_idx, target_idx = batch_loader.next_embedding_seq(args.batch_size)

        input = Variable(t.from_numpy(input_idx).long())
        target = Variable(t.from_numpy(target_idx).long())
        if args.use_cuda:
            input, target = input.cuda(), target.cuda()

        out = neg_loss(input, target, args.num_sample).mean()

        optimizer.zero_grad()
        out.backward()
        optimizer.step()

        if iteration % 500 == 0:
            out = out.cpu().data.numpy()[0]
            print('iteration = {}, loss = {}'.format(iteration, out))

    word_embeddings = neg_loss.input_embeddings()
    np.save('data/preprocessings/word_embeddings.npy', word_embeddings)