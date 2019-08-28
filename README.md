# MNIST-neural-network-classifier
A trained neural network built from scratch with architect [784, 3000, 2500, 2000, 1500, 1000, 500, 10] to classifier clothes and digits from the MNIST dataset. You can modify the neural network structure or re-train on your own dataset.

Training set accuracy on MNIST handwritten digits: 99.9%

Test set error on MNIST handwritten digits: 98%

Download the source code file and run "gradient_descent" to train and "demo" to test on a random image in the test set

To verify the performance of the algorithm:
print(set_performance(X_train, Y_train, W, b));
print(set_performance(X_test, Y_test, W, b))

To change the saved weights file path, modify the following variable to your own desired path:
weights_file = './Kien/mnist_classifier/fashion_mnist_trained_weights_deep.dat'

You can also play around with the code and try different network structure, algorithms, or training sets. Have fun!
