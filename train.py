import argparse
import os
import numpy as np
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import RMSprop
from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rgan import RGAN

if __name__ == "__main__":

    if not os.path.exists('data/preprocessings/word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    parser = argparse.ArgumentParser(description='RGAN')
    parser.add_argument('--num-iterations', type=int, default=65000, metavar='NI',
                        help='num iterations (default: 65000)')
    parser.add_argument('--batch-size', type=int, default=60, metavar='BS',
                        help='batch size (default: 60)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    args = parser.parse_args()

    batch_loader = BatchLoader('')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.vocab_size)

    rgan = RGAN(parameters)
    if args.use_trained:
        rgan.load_state_dict(t.load('trained_RGAN'))
    if args.use_cuda:
        rgan = rgan.cuda()

    g_optimizer = RMSprop(rgan.generator.parameters(), args.learning_rate)
    d_optimizer = RMSprop(rgan.discriminator.parameters(), args.learning_rate)

    for iteration in range(args.num_iterations):

        for _ in range(2):
            '''Dicriminator forward-loss-backward-update'''
            z, true_data = batch_loader.input_data(args.batch_size, args.use_cuda, parameters)

            discriminator_loss, _ = rgan(z, true_data)

            d_optimizer.zero_grad()
            discriminator_loss.backward()
            d_optimizer.step()

            for p in rgan.discriminator.parameters():
                p.data.clamp_(-0.05, 0.05)

        '''Generator forward-loss-backward-update'''
        z, true_data = batch_loader.input_data(args.batch_size, args.use_cuda, parameters)

        discriminator_loss, generator_loss = rgan(z, true_data)

        g_optimizer.zero_grad()
        generator_loss.backward()
        g_optimizer.step()

        if iteration % 1 == 0:
            print('------------------------------------')
            print('iteration {}'.format(iteration))
            print('-----------GENERATOR LOSS-----------')
            print(generator_loss.data.cpu().numpy()[0])
            print('---------DISCRIMINATOR LOSS---------')
            print(discriminator_loss.data.cpu().numpy()[0])

            '''Sample data from z'''
            z = batch_loader.sample_z(batch_size=1, use_cuda=args.use_cuda, params=parameters)

            sampling = rgan.sample(z, 35, batch_loader)
            print(sampling)

    t.save(rvae.state_dict(), 'trained_RGAN')
