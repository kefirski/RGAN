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


def chain(input, *args):
    """
    Applies to input functions from *args
    """

    for arg in args:
        input = arg(input)

    return input