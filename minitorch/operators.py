"""
Collection of the core mathematical operators used throughout the code base.
"""

import math


# ## Task 0.1

# Implementation of a prelude of elementary functions.


def mul(x, y):
    ":math:`f(x, y) = x * y`"
    return x * y


def id(x):
    ":math:`f(x) = x`"
    return x


def add(x, y):
    ":math:`f(x, y) = x + y`"
    return x + y


def neg(x):
    ":math:`f(x) = -x`"
    return - x


def lt(x, y):
    ":math:`f(x) =` 1.0 if x is less than y else 0.0"
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x, y):
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x, y):
    ":math:`f(x) =` x if x is greater than y else y"
    if x > y:
        return x
    else:
        return y


def is_close(x, y):
    ":math:`f(x) = |x - y| < 1e-2` "
    return lt(abs(add(x, neg(y))), 1e-2)


def sigmoid(x):
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    Args:
        x (float): input

    Returns:
        float : sigmoid value
    """
    if lt(0, x):
        return 1.0 / add(1.0, exp(neg(x)))
    else:
        return exp(x) / add(1.0, exp(x))


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    if x >= 0:
        return x
    else:
        return 0


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    if x > 0:
        return math.log(x)
    else:
        raise ValueError


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(x, d):
    r"If :math:`f = log` as above, compute d :math:`d \times f'(x)`"
    return d / x


def inv(x):
    ":math:`f(x) = 1/x`"
    if x != 0:
        return 1 / x
    else:
        raise ZeroDivisionError


def inv_back(x, d):
    r"If :math:`f(x) = 1/x` compute d :math:`d \times f'(x)`"
    return neg(d / x**2)


def relu_back(x, d):
    r"If :math:`f = relu` compute d :math:`d \times f'(x)`"
    if x > 0:
        return d
    elif x == 0:
        return EPS
    else:
        return 0.0


# ## Task 0.3

# Small library of elementary higher-order functions for practice.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    def apply(ls):
        ret = []
        for elem in ls:
            ret.append(fn(elem))
        return ret
    return apply


def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    neg_list_func = map(neg)
    return neg_list_func(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """

    def apply(ls1, ls2):
        ret = []
        for elem1, elem2 in zip(ls1, ls2):
            ret.append(fn(elem1, elem2))
        return ret
    return apply


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    add_lists_func = zipWith(add)
    return add_lists_func(ls1, ls2)


def reduce(fn, start=None):
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """
    def apply(ls):
        it = iter(ls)
        if start is None:
            value = next(it)
        else:
            value = start
        for element in it:
            value = fn(value, element)
        return value
    return apply


INITIAL_VALUE_SUM = 0
INITIAL_VALUE_MUL = 1


def sum(ls):
    "Sum up a list using :func:`reduce` and :func:`add`."
    sum_func = reduce(add, INITIAL_VALUE_SUM)
    return sum_func(ls)


def prod(ls):
    "Product of a list using :func:`reduce` and :func:`mul`."
    sum_func = reduce(mul, INITIAL_VALUE_MUL)
    return sum_func(ls)

