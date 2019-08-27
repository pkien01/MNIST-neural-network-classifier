import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

(X_train, label_train), (X_test, label_test) = tf.keras.datasets.fashion_mnist.load_data()

X_test_images = X_test
X_train_images = X_train
#print('Train: X=%s, y=%s' % (X_train.shape, Y_train.shape))
#print('Test: X=%s, y=%s' % (X_test.shape, Y_test.shape))

m_train = X_train.shape[0]
m_test = X_test.shape[0]
n_x = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0], -1).T / 255
X_test = X_test.reshape(X_test.shape[0], -1).T / 255

Y_train = np.zeros((m_train, 10))
Y_train[np.arange(m_train), label_train] = 1
Y_train = Y_train.T

Y_test = np.zeros((m_test, 10))
Y_test[np.arange(m_test), label_test] = 1
Y_test = Y_test.T

#Network structure
n_classes = 10
#n = [n_x, 2500, 2000, 1500, 1000, 500, n_classes]
#n = [n_x, 800, n_classes]
n = [n_x, 3000, 2500, 2000, 1500, 1000, 500, n_classes]

L = len(n)
Wfilename = './Kien/mnist_classifier/fashion_mnist_trained_weights_deep.dat'

def relu(x):
    return np.maximum(0, x)
    
def softmax(x):
    t = np.exp(x - np.max(x, axis = 0).reshape((1, x.shape[1])))
    return t / np.sum(t, axis = 0, keepdims = True)

def forward_prop(X, W, b):
    m = X.shape[1]
    Z = [None]
    A = [X]
    for l in range(1, L):
        Z.append(np.dot(W[l], A[l - 1]) + b[l])
        if (l == L - 1): A.append(softmax(Z[l]))
        else: A.append(relu(Z[l]))  
        assert(Z[l].shape == (n[l], m))
        assert(A[l].shape == Z[l].shape)
    return Z, A

import pickle    
import os

def load_para():
    file_exists = os.path.isfile(Wfilename)
    if (file_exists):
        Wfile = open(Wfilename, 'rb')
        W, b = pickle.load(Wfile)
        Wfile.close()
    else:
        W = [None]
        b = [None]
        for l in range(1, L):
            #np.sqrt(2./ n[l])
            W.append(np.random.randn(n[l], n[l - 1]) * np.sqrt(2. / n[l - 1]))
            b.append(np.zeros((n[l], 1)))
        Wfile = open(Wfilename, 'wb')
        pickle.dump([W, b], Wfile)
        Wfile.close()
    return W, b
    

def split_batches(X, Y, batch_size):
    m = X.shape[1]
    assert(m == Y.shape[1])
    perm = list(np.random.permutation(m))
    shuffled_X = X[:, perm]
    shuffled_Y = Y[:, perm].reshape((n_classes, m))
    assert(shuffled_X.shape == X.shape)
    assert(shuffled_Y.shape == Y.shape)
    n_batches = m // batch_size
    batches = []
    for i in range(0, n_batches):
        batch_X = shuffled_X[:, i * batch_size : (i + 1) * batch_size]
        batch_Y = shuffled_Y[:, i * batch_size : (i + 1) * batch_size]
        batches.append((batch_X, batch_Y))
    if (m % batch_size != 0):
        batch_X = shuffled_X[:, batch_size * n_batches : m]
        batch_Y = shuffled_Y[:, batch_size * n_batches : m]
        batches.append((batch_X, batch_Y))
    return batches

def set_performance(X, Y, W, b, batch_size = 2**8):
    m = X.shape[1]
    assert(m == Y.shape[1])
    batches = split_batches(X, Y, batch_size) 
    acc = 0
    for batch_idx in range(len(batches)):
        X_cur, Y_cur = batches[batch_idx]
        m_cur = X_cur.shape[1]
        assert(m_cur == Y_cur.shape[1])
        _, A_cur = forward_prop(X_cur, W, b)
        pred = np.argmax(A_cur[L - 1], axis = 0).reshape((m_cur, 1))
        label = np.argmax(Y_cur, axis = 0).reshape((m_cur, 1))
        acc += np.sum(pred == label)
    return acc / m

fashion_type = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
import random
def demo(W, b, idx = random.randint(0, m_test - 1), fashion = False):
    _, A = forward_prop(X_test[:, idx].reshape(n_x, -1), W, b) 
    pred = np.squeeze(np.argmax(A[L - 1]))
    sample = X_test_images[idx]
    plt.imshow(sample)
    if (fashion): plt.suptitle("Prediction label: " + fashion_type[pred] + "  |  Ground truth label: " + fashion_type[label_test[idx]] )  
    else: plt.suptitle("Prediction label: " + str(pred) + "  |  Ground truth label: " + str(label_test[idx]) )
    plt.title("Confidence: " + str(np.squeeze(A[L - 1][pred]) * 100.) + "%.")
    plt.show()

def demo_wrong(W, b, fashion = False):
    while (True):
        i = random.randint(0, m_test - 1)
        _, A = forward_prop(X_test[:, i].reshape(n_x, -1), W, b) 
        pred = np.squeeze(np.argmax(A[L - 1]))
        truth = label_test[i]
        if (pred != truth):
            demo(W, b, i, fashion)
            break

W, b = load_para()

print(set_performance(X_train, Y_train, W, b))
print(set_performance(X_test, Y_test, W, b))


#demo(W, b, fashion = True)