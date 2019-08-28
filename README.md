# MNIST-neural-network-classifier
A trained neural network built from scratch with architecture [784, 3000, 2500, 2000, 1500, 1000, 500, 10] (ReLU on all hidden layers and softmax on the output layer) to classify clothes and digits from the MNIST dataset. You can modify the neural network structure or re-train on your own dataset.

Training set accuracy on MNIST handwritten digits: 99.9%

Test set error on MNIST handwritten digits: 98%

Download the source code file and run "gradient_descent" to train, "demo" to visualize the prediction on a random image in the test set, and "demo_wrong" to visualize the predicition on a random image that the algorithm mislabeled on the test set.

To train on a different datset modify the following line:
(X_train, label_train), (X_test, label_test) = tf.keras.datasets.fashion_mnist.load_data()

For example: (X_train, label_train), (X_test, label_test) = tf.keras.datasets.mnist.load_data() #handwritten digits dataset

To verify the performance of the algorithm:
print(set_performance(X_train, Y_train, W, b));
print(set_performance(X_test, Y_test, W, b))

To change the saved weights file path, modify the following variable to your own desired path:
weights_file = './Kien/mnist_classifier/fashion_mnist_trained_weights_deep.dat'

To tune the hyperparameters, modify the "gradient_descent" input variables: keep_prob = the probability of keeping the activation units in dropout, lbd = L2 regularization parameter, learning_rate, batch_size, n_iters = number of iterations/epochs over the training set, beta1 and beta2 = parameters for Adam optimization algorithm, decay_rate = learning rate decay rate over each epoch. 

You can also play around with the code and try out different network architechture, algorithms, or data sets. Have fun!
