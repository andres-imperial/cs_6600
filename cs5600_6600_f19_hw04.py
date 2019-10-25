# /usr/bin/python

################################################################
# Andres Imperial
# Your 02294771
# Write your code at the end of this file in the provided
# function stubs.
#
# Note: Put parens around print statements if you're using Py3.
################################################################

# Libraries
# Standard library
import json
import random
import sys
import os

# Third-party libraries
import numpy as np
#import scipy as sp
#import matplotlib.pyplot as plt
import cv2

import pickle as cPickle
import network2

# Globals
saveDir = "/home/aimperial/School/cs_6600/hw04_f19/pickle_nets/"
train_d, valid_d, test_d = [],[],[]

# save() function to save the trained network to a file


def save(ann, file_name):
    with open(file_name, 'wb') as fp:
        cPickle.dump(ann, fp)

# restore() function to restore the file


def load(file_name):
    with open(file_name, 'rb') as fp:
        nn = cPickle.load(fp)
    return nn


# Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5 * np.linalg.norm(a - y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a - y)


# Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.init_weights`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.init_weights()
        self.cost = cost

    # normalized weight initializer
    def sqrt_norm_init_weights(self):
        """Initialize random weights with a standard deviation of 1/sqrt(x).
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    # large weight initializer
    def init_weights(self):
        """Initialize random weights.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        if evaluation_data:
            n_eval_data = len(evaluation_data)
        n_train_data = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n_train_data, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print ("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print ("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy / float(n_train_data))
                print ("Accuracy on training data: {} / {}".format(
                    accuracy, n_train_data))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print ("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy / float(n_eval_data))
                print ("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_eval_data))

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    # vladimir kulyukin 14may2018: same as above but
    # the accuracy function is called with convert=True always
    # to accomodate the bee data.
    def SGD2(self, training_data, epochs, mini_batch_size, eta,
             lmbda=0.0,
             evaluation_data=None,
             monitor_evaluation_cost=False,
             monitor_evaluation_accuracy=False,
             monitor_training_cost=False,
             monitor_training_accuracy=False):
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            # print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print ("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print ("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print ("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, convert=True)
                evaluation_accuracy.append(accuracy)
                # vladimir kulyukin: commented out
                # print "Accuracy on evaluation data: {} / {}".format(
                #    accuracy, n)
            # print
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass: zs[-1] is not used.
        # activations[-1] - y = (a - y).
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        # delta = (a^{L}_{j} - y_{j})
        nabla_b[-1] = delta
        # nabla_w = a^{L-1}_{k}(a^{L}_{j} - y_{j}).
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

# Loading a Network


def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

# Miscellaneous functions


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def plot_costs(eval_costs, train_costs, num_epochs):
    # your code here
    plt.plot(range(0, num_epochs), eval_costs, 'r',
             range(0, num_epochs), train_costs, 'b')
    plt.suptitle("Evaluation cost (red) and Training cost (blue)")
    plt.xlabel('epochs')
    plt.ylabel('costs')
    plt.show()
    pass


def plot_accuracies(eval_accs, train_accs, num_epochs):
    # your code here
    plt.plot(range(0, num_epochs), eval_accs, 'r',
             range(0, num_epochs), train_accs, 'b')
    plt.suptitle("Evaluation accuracies (red) and Training accuracies (blue)")
    plt.xlabel('epochs')
    plt.ylabel('accuracies')
    plt.show()
    pass

# num_nodes -> (eval_cost, eval_acc, train_cost, train_acc)
# use this function to compute the eval_acc and min_cost.


def save_and_plot(net_stats_dict, myNet, lmbda, eval_data, train_data, epochs):
    # Make filename
    currentPath = saveDir + "net"
    for size in myNet.sizes:
        currentPath += "_" + str(size)

    currentPath += "_" + str(int(eta * 100))
    currentPath += "_" + str(mbs) + ".pck"
    save(myNet.weights, currentPath)

    # plot stuff
    plot_costs(net_stats_dict[lmbda][0], net_stats_dict[lmbda][2], epochs)
    lA = [float(x) / len(eval_data) for x in net_stats_dict[lmbda][1]]
    lB = [float(x) / len(train_data) for x in net_stats_dict[lmbda][3]]
    plot_accuracies(lA, lB, epochs)


# My best net for 1 layer was with a structure of 94 nodes in the hidden layer
# and achieved 97.64% accuracy over 30 epochs with a lambda value of 2
def collect_1_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     cost_function,
                                     epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data):
    net_stats_dict = {}
    for num_nodes in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 1):
        myNet = network2.Network([784, num_nodes, 10], cost_function)
        net_stats_dict[num_nodes] = myNet.SGD(train_d, epochs, mbs, eta, lmbda,
                                              evaluation_data=valid_d,
                                              monitor_evaluation_cost=True,
                                              monitor_evaluation_accuracy=True,
                                              monitor_training_cost=True,
                                              monitor_training_accuracy=True)

    return net_stats_dict


# My best net for 2 layers was with a structure of 90, 98 and achieved 97.89%
# accuracy over 30 epochs with a lambda value of 2
def collect_2_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     cost_function,
                                     epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data):
    net_stats_dict = {}
    for n1 in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 1):
        for n2 in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 1):
            myNet = network2.Network([784, n1, n2, 10], cost_function)
            net_stats_dict[str(n1) + "_" + str(n2)] = myNet.SGD(train_d, epochs, mbs, eta, lmbda,
                                                                evaluation_data=valid_d,
                                                                monitor_evaluation_cost=True,
                                                                monitor_evaluation_accuracy=True,
                                                                monitor_training_cost=True,
                                                                monitor_training_accuracy=True)

    return net_stats_dict


# My best net for 3 layers was with a structure of 79, 39, 45 and achieved 97.67%
# accuracy over 30 epochs with a lambda value of 4
def collect_3_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     cost_function,
                                     epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data):
    net_stats_dict = {}
    for n1 in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 1):
        for n2 in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 1):
            for n3 in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 1):
                myNet = network2.Network([784, n1, n2, n3, 10], cost_function)
                net_stats_dict[str(n1) + "_" + str(n2) + "_" + str(n3)] = myNet.SGD(train_d, epochs, mbs, eta, lmbda,
                                                                                    evaluation_data=valid_d,
                                                                                    monitor_evaluation_cost=True,
                                                                                    monitor_evaluation_accuracy=True,
                                                                                    monitor_training_cost=True,
                                                                                    monitor_training_accuracy=True)

    return net_stats_dict


def runTrails():
    stats_for_one_layer = {}
    for lmbda in range(0, 5, 1):
        stats_for_one_layer.update(collect_1_hidden_layer_net_stats(30, 100, CrossEntropyCost, 30, 10, 0.1, lmbda,
                                                                    train_d, test_d))
    print (min(stats_for_one_layer.items(), key=lambda x: x[1][0][-1]))

    stats_for_two_layers = {}
    for lmbda in range(0, 5, 1):
        stats_for_two_layers.update(collect_2_hidden_layer_net_stats(30, 100, CrossEntropyCost, 30, 10, 0.1, lmbda,
                                                                     train_d, test_d))
    print (min(stats_for_two_layers.items(), key=lambda x: x[1][0][-1]))

    stats_for_three_layers = {}
    for lmbda in range(0, 5, 1):
        stats_for_three_layers = collect_3_hidden_layer_net_stats(30, 100, CrossEntropyCost, 30, 10, 0.1, lmbda,
                                                                  train_d, test_d)
    print (min(stats_for_three_layers.items(), key=lambda x: x[1][0][-1]))

    print (min(stats_for_one_layer.items(), key=lambda x: x[1][0][-1]))
    print (min(stats_for_two_layers.items(), key=lambda x: x[1][0][-1]))
    print (min(stats_for_three_layers.items(), key=lambda x: x[1][0][-1]))
    pass


#====================================
def convertAndScaleBeeImage(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image/255.0

    return scaled_gray_image

def loadBeeImage(imagePath):
    img = cv2.imread(imagePath)

    return img

def loadBeeImages(directory):
    beeData = []

    
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".png"):
                img = cv2.imread(subdir + os.sep + filename)
                img = convertAndScaleBeeImage(img)
                beeData.append(img)
                continue
            else:
                continue

#   for filename in os.listdir(directory):
#       if filename.endswith(".png"):
#           img = cv2.imread(directory + filename)
#           img = convertAndScaleBeeImage(img)
#           beeData.append(img)
#           continue
#       else:
#           continue

    return beeData

def getBeeTestingData():
    #=========================================================
    beeImages = loadBeeImages("/home/Ders/Documents/CS_5600/project_1/data/BEE1/bee_test")
    expected = np.ndarray(shape = (2, 1), dtype=int)
    expected[0] = [1]
    expected[1] = [0]

    trainingList = []
    for index in range(len(beeImages)):
        flattenedBeeData = beeImages[index].flatten()
        flattenedBeeData = flattenedBeeData.reshape(len(flattenedBeeData), 1)
        trainingList.append((flattenedBeeData, expected))
    #=========================================================

    #=========================================================
    noBeeImages = loadBeeImages("/home/Ders/Documents/CS_5600/project_1/data/BEE1/no_bee_test/")
    expected = np.ndarray(shape = (2, 1), dtype=int)
    expected[0] = [0]
    expected[1] = [1]

    noTrainingList = []
    for index in range(len(noBeeImages)):
        flattenedBeeData = noBeeImages[index].flatten()
        flattenedBeeData = flattenedBeeData.reshape(len(flattenedBeeData), 1)
        noTrainingList.append((flattenedBeeData, expected))
    #=========================================================

    for sample in noTrainingList:
        trainingList.append(sample)

    random.shuffle(trainingList)

    return trainingList
    pass
def getBeeTrainingData():
    #=========================================================
    beeImages = loadBeeImages("/home/Ders/Documents/CS_5600/project_1/data/BEE1/bee_train/")
    expected = np.ndarray(shape = (2, 1))
    expected[0] = [1]
    expected[1] = [0]

    trainingList = []
    for index in range(len(beeImages)):
        flattenedBeeData = beeImages[index].flatten()
        flattenedBeeData = flattenedBeeData.reshape(len(flattenedBeeData), 1)
        trainingList.append((flattenedBeeData, expected))
    #=========================================================

    #=========================================================
    noBeeImages = loadBeeImages("/home/Ders/Documents/CS_5600/project_1/data/BEE1/no_bee_train/")
    expected = np.ndarray(shape = (2, 1))
    expected[0] = [0]
    expected[1] = [1]

    noTrainingList = []
    for index in range(len(noBeeImages)):
        flattenedBeeData = noBeeImages[index].flatten()
        flattenedBeeData = flattenedBeeData.reshape(len(flattenedBeeData), 1)
        noTrainingList.append((flattenedBeeData, expected))
    #=========================================================

    for sample in noTrainingList:
        trainingList.append(sample)

    random.shuffle(trainingList)

    return trainingList
    pass

def testTrain():
    trainData = getBeeTrainingData()
    trainData = trainData[:10000]
    testData = getBeeTestingData()
    testData = testData[:5000]

    print(testData)

    myNet = network2.Network([1024, 128, 32, 2], CrossEntropyCost)
    net_stats = myNet.SGD(trainData, 50, 30, 0.15, 0.1,
                          evaluation_data=testData,
                          monitor_evaluation_cost=True,
                          monitor_evaluation_accuracy=True,
                          monitor_training_cost=True,
                          monitor_training_accuracy=True)


#====================================

testTrain()
#runTrails()
