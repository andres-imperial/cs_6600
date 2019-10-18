#!/usr/bin/python

####################################################
# Name: Andres Imperial
# 
# module: mnist_digits_random_forest.py
# description: Testing random forest for MNIST datasets
# bugs to vladimir dot kulyukin at usu dot edu
####################################################

from sklearn import tree, metrics
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from mnist_loader import load_data_wrapper

mnist_train_data, mnist_test_data, mnist_valid_data = \
                  load_data_wrapper()

mnist_train_data_dc = np.zeros((50000, 784))
mnist_test_data_dc  = np.zeros((10000, 784))
mnist_valid_data_dc = np.zeros((10000, 784))

mnist_train_target_dc = None
mnist_test_target_dc  = None
mnist_valid_target_dc = None

def reshape_mnist_aux(mnist_data, mnist_data_dc):
    '''auxiliary function to reshape MNIST data for sklearn.'''
    for i in range(len(mnist_data)):
        mnist_data_dc[i] = mnist_data[i][0].reshape((784,))

def reshape_mnist_data():
    '''reshape all MNIST data for sklearn.'''
    global mnist_train_data
    global mnist_train_data_dc
    global mnist_test_data
    global mnist_test_data_dc
    global mnist_valid_data
    global mnist_valid_data_dc
    reshape_mnist_aux(mnist_train_data, mnist_train_data_dc)
    reshape_mnist_aux(mnist_test_data,  mnist_test_data_dc)
    reshape_mnist_aux(mnist_valid_data, mnist_valid_data_dc)

def reshape_mnist_target(mnist_data):
    '''reshape MNIST target given data.'''
    return np.array([np.argmax(mnist_data[i][1])
                    for i in range(len(mnist_data))])

def reshape_mnist_target2(mnist_data):
    '''another function for reshaping MNIST target given data.'''
    return np.array([mnist_data[i][1] for i in range(len(mnist_data))])

def prepare_mnist_data():
    '''reshape and prepare MNIST data for sklearn.'''
    global mnist_train_data
    global mnist_test_data
    global mnist_valid_data
    reshape_mnist_data()

    ### make sure that train, test, and valid data are reshaped
    ### correctly.
    for i in range(len(mnist_train_data)):
        assert np.array_equal(mnist_train_data[i][0].reshape((784,)),
                              mnist_train_data_dc[i])

    for i in range(len(mnist_test_data)):
        assert np.array_equal(mnist_test_data[i][0].reshape((784,)),
                              mnist_test_data_dc[i])

    for i in range(len(mnist_valid_data)):
        assert np.array_equal(mnist_valid_data[i][0].reshape((784,)),
                              mnist_valid_data_dc[i])

def prepare_mnist_targets():
    '''reshape and prepare MNIST targets for sklearn.'''
    global mnist_train_target_dc
    global mnist_test_target_dc
    global mnist_valid_target_dc    
    mnist_train_target_dc = reshape_mnist_target(mnist_train_data)
    mnist_test_target_dc  = reshape_mnist_target2(mnist_test_data)
    mnist_valid_target_dc = reshape_mnist_target2(mnist_valid_data)    

def test_rf(num_trees):
    ## your code here
    rs = random.randint(0, 1000)
    clf = RandomForestClassifier(n_estimators=num_trees,
    random_state=rs)
    rf = clf.fit(mnist_train_data_dc, mnist_train_target_dc)
    valid_preds = rf.predict(mnist_valid_data_dc)
    #print(metrics.classification_report(mnist_valid_target_dc,
    #valid_preds))

    cm1 = confusion_matrix(mnist_valid_target_dc,
    valid_preds)
    #print(cm1)
    #print(metrics.accuracy_score(mnist_valid_target_dc, valid_preds))

    return (metrics.accuracy_score(mnist_valid_target_dc, valid_preds), cm1)

def test_rf_range(low_nt, high_nt):
    ## your code here
    topThree = [0,0,0]
    models = {}
    for x in range(low_nt, high_nt):
        accuracy, model = test_rf(x)
        if min(topThree) < accuracy:
            topThree.append(accuracy)
            models[accuracy] = model
            topThree.remove(min(topThree))
            
    for x in topThree:
        print("\n", x, models[x], "\n")

    print(models, "\n")

    pass

def test_dt():
    ## your code here
    clf = tree.DecisionTreeClassifier(random_state=0)
    dtr = clf.fit(mnist_train_data_dc, mnist_train_target_dc)
    print ("training completed")
    valid_preds = dtr.predict(mnist_valid_data_dc)

    print(metrics.classification_report(mnist_valid_target_dc,
    valid_preds))

    cm1 = confusion_matrix(mnist_valid_target_dc, valid_preds)
    print(cm1)

    pass

if __name__ == '__main__':
    prepare_mnist_data()
    prepare_mnist_targets()
    #test_dt()    
    #test_rf(5)
    test_rf_range(5, 10)
    


