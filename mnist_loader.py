#!/usr/bin/python


import pickle
import gzip
import numpy as np


# To load raw data
# >>> trd, vad, ted = load_data()
# trd[0] are images
# trd[1] are classifications
# trd[0][0].shape --> (784,)
# trd[1][0] --> 5

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data


# trd is the training data, vad is the validiation data,
# ted is the test data.
# trd consists of 50,000 images each of which has 784 pixels,
# i.e., 28*28 = 784; trd[1] consists of 50,000 digits;
# vad[0] consists of 10,000 images each of which has 784 pixels;
# vad[1] has 10,000 digits. 
# >>> trd, vad, ted = load_data()
# >>> len(trd)
# 2
# >>> len(vad)
# 2
# >>> len(ted)
# 2
# >>> trd1, vad1, ted1 = load_data_wrapper()
# >>> len(trd1)
# 50000
# trd[1][0] --> 5
# trd1[0][1] --> array([[ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 1.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.]])
# >>> trd[1][1]
# 0
# >>> trd1[1][1]
# array([[ 1.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.]])
# >>> trd[1][2]
# 4
# >>> trd1[2][1]
# array([[ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 1.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.]])
# >>> trd1[2][1]
# array([[ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 1.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.]])
# >>> trd[1][3]
# 1
# >>> trd1[3][1]
# array([[ 0.],
#       [ 1.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.]])

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)


# >>> vectorized_result(5)
# array([[ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 1.],
#       [ 0.],
#       [ 0.],
#       [ 0.],
#       [ 0.]])
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
