def fold(f, l, a):
    return a if (len(l) == 0) else fold(f, l[1:], f(a, l[0]))


def f_and(x, y):
    return x and y


def f_or(x, y):
    return x or y


def parameters_allocation_check(module):

    parameters = list(module.parameters())
    parameters = [param.is_cuda for param in parameters]

    return fold(f_and, parameters, True) or not fold(f_or, parameters, False)


def input_data(batch_loader, batch_size, use_cuda):
    import torch as t
    from torch.autograd import Variable

    true_data = batch_loader.true_data(batch_size, 'train')
    true_data = Variable(t.from_numpy(true_data)).long()
    if use_cuda:
        true_data = true_data.cuda()

    z = Variable(t.rand([args.batch_size, params.latent_variable_size]))

    return z, true_data