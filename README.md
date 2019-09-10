# MNIST Neural Network classifier 

## The model

This is a trained neural network built from scratch to classify handwritten digits or fashions from the MNIST dataset. 

By default, the neural network has an architecture:

*L* = number of layers = 8

| Layer index (*l*) | Number of Activation units (*n[l]*) | Activation function |
| ----------------- | ----------------------------------- | ------------------- |
| 0                 | 784                                 | N/A                 |
| 1                 | 3000                                | ReLU                |
| 2                 | 2500                                | ReLU                |
| 3                 | 2000                                | ReLU                |
| 4                 | 1500                                | ReLU                |
| 5                 | 1000                                | ReLU                |
| 6                 | 500                                 | ReLU                |
| 7                 | 10                                  | Softmax             |

Trained using Adam optimizer and initialized parameters with He-et-al initialization.

Default hyperparameters settings:

| Hyperparameter name                                   | Value |
| ----------------------------------------------------- | ----- |
| Learning rate (alpha)                                 | 0.001 |
| Number of epochs (n_epochs)                           | 2000  |
| Batch size (batch_size)                               | 256   |
| Dropout keep activation units probability (keep_prob) | 0.7   |
| L2 regularization parameter (lbd)                     | 0.05  |
| Adam's parameter β1 (beta1)                           | 0.9   |
| Adam's parameter β2 (beta2)                           | 0.999 |
| Learning rate decay (decay_rate)                      | 1.0   |



## Installation, demo, and training

**Prerequisites**: Make sure you've installed the required libraries/packages: Numpy, Tensorflow (only used to get the dataset), and Matplotlib

1. Clone this repository 

   ```shell
   git clone https://github.com/pkien01/MNIST-neural-network-classifier
   ```

2. Download the pretrained weights for the MNIST digits and fashions dataset [here](https://drive.google.com/drive/folders/1CmQRokKnQ75ukEU_Y5Lq9DsjYVWxP6MM?usp=sharing) and move it to the `MNIST-neural-network-classifier` folder.

3. After that,  the `MNIST-neural-network-classifier` folder should have the following structure

   ```bash
   MNIST-neural-network-classifier/                         
   MNIST-neural-network-classifier/model.py #the model source code
   MNIST-neural-network-classifier/mnist_trained_weights_deep.dat #the pretrained MNIST digits weights
   MNIST-neural-network-classifier/fashion_mnist_trained_weights_deep.dat #the pretrained MNIST fashion weights
   ```

4. Open the `model.py` source code file and at the end, it should be similar to the following 

   ```python
   ...
   W, b = load_cache()
   
   #gradient_descent(W, b, keep_prob=.7, lbd =.03, learning_rate=0.001)
   #print(set_performance(X_train, Y_train, W, b))
   #print(set_performance(X_test, Y_test, W, b))
   demo(W, b, fashion=False)
   #demo_wrong(W, b, fashion=False)
   #demo_all_layers(W, b)
   ```

5. Run the source code file

   ```bash
   cd MNIST-neural-network-classifier
   python3 model.py
   ```

   Congrats! You've just ran the demo on the MNIST handwritten digits dataset

6. To run it on the MNIST fashion dataset:

   - Open the `model.py` in a text/code editor

   - Change the following line (line 8 in the default source code)

     ```python
     (X_train, label_train), (X_test, label_test) = tf.keras.datasets.mnist.load_data()
     ```

     to this

     ```
     (X_train, label_train), (X_test, label_test) = tf.keras.datasets.fashion_mnist.load_data()
     ```

   - Next, change the following line (line 24 in the default source code)

     ```python
     weights_file = './MNIST-neural-network-classifier/mnist_trained_weights_deep.dat'
     ```

     to this

     ```python
     weights_file = './MNIST-neural-network-classifier/fashion_mnist_trained_weights_deep.dat'
     ```

   * Next, set the `fashion` variable to `True`  in the `demo` function

     ```python
     demo(W, b, fashion=False)
     #demo_wrong(W, b, fashion=False)
     ```

     to this

     ```python
     demo(W, b, fashion=True)
     #demo_wrong(W, b, fashion=True)
     ```
   * Repeat step 5 to run the source code file

  7. To demo on incorrectly labeled images, uncomment the line `demo_wrong(W, b, fashion=False)` and comment all the other function calls (of course, the `fashion`variable can be set `True` or `False` depending if you want to demo on the fashion or the handwritten digits images). Then, repeat step 5 to run the source code.

  8. To verify the model performance on train and test set, respectively, uncomment the following lines (and comment all the other function calls).  

     ```python
     print(set_performance(X_train, Y_train, W, b))
     print(set_performance(X_test, Y_test, W, b))
     ```

     Repeat step 5 to run the source code

  9. To visualize each of the individual layers' activations, uncomment the function `demo_all_layers(W, b)` and comment the rest function calls. Repeat step 5 to run the source code.

  10. To re-train it from the trained weights, uncomment the function `gradient_descent(W, b, keep_prob=.7, lbd =.03, learning_rate=0.001)` and comment the rest function calls. You can use your own set of hyperparameters to train the neural network or tune it yourself if you want. 

      If you want to train from scratch, delete the weights file first before training (though it may take quite a long time to train): `mnist_trained_weights_deep.dat` for the MNIST handwritten digits dataset and `fashion_mnist_trained_weights_deep.dat` for the MNIST fashion dataset. 

      You can also train it using a different algorithm like standard gradient descent or gradient descent with momentum, instead of Adam, by modifying the initialization and parameters update function calls in the `gradient descent()` function. For example, the update function of standard gradient descent is `update_para(W, b, dW, db, learning_rate)` and gradient descent with momentum is `update_para_momentum(W, b, dW, db, VdW, Vdb, epoch_num, learning_rate, beta1)`. 



## Results of pretrained weights

* On the MNIST handwritten digits dataset

  Training set accuracy: 99.99833%.

  Test set accuracy: 98.09%.

* On the MNIST fashion dataset

  Training set accuracy: 99.39%.

  Test set accuracy: 89.24%.
  

You can train it for a longer period,  and/or adjust the hyperparameters, to get better performance.



Here are the results on some example images on the handwritten digits dataset:

<p align="middle">
	<img src="https://raw.githubusercontent.com/pkien01/MNIST-neural-network-classifier/master/sample_images/3.png" style="zoom:60%" /> 
	<img src="https://raw.githubusercontent.com/pkien01/MNIST-neural-network-classifier/master/sample_images/4.png" style="zoom:60%" /> 
</p>
<p align="middle">
	<img src="https://raw.githubusercontent.com/pkien01/MNIST-neural-network-classifier/master/sample_images/9.png" style="zoom:60%" /> 
	<img src="https://raw.githubusercontent.com/pkien01/MNIST-neural-network-classifier/master/sample_images/0_wrong.png" style="zoom:60%" />
</p>	

And here are some on the fashion dataset:

<div class="row">
	<div class="column">
		<img src="https://raw.githubusercontent.com/pkien01/MNIST-neural-network-classifier/master/sample_images/shirt.png" style="zoom:60%" />
		<img src="https://raw.githubusercontent.com/pkien01/MNIST-neural-network-classifier/master/sample_images/trouser.png" style="zoom:60%" />
	</div>
	<div class="column">
		<img src="https://raw.githubusercontent.com/pkien01/MNIST-neural-network-classifier/master/sample_images/ankle_boot.png" style="zoom:60%" />
		<img src="https://raw.githubusercontent.com/pkien01/MNIST-neural-network-classifier/master/sample_images/bag_wrong.png" style="zoom:60%" />
	</div>
</div>

Finally, here are the visualizations of all the layers' activations (looks pretty random, huh?):
![](https://raw.githubusercontent.com/pkien01/MNIST-neural-network-classifier/master/sample_images/all_layers.png)
![](https://raw.githubusercontent.com/pkien01/MNIST-neural-network-classifier/master/sample_images/all_layers_fashion.png)

Have fun playing around with the model as you like.



## Contacts

If you have any questions or encounter some serious errors/bugs in the code, you can write an email to phamkienxmas2001@gmail.com, v.kienp16@vinai.io, or kindly leave a message below on GitHub. Thank you!