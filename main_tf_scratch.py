
"""
DESRIPTION:
In this script I code from scratch the solution proposed
in main_tf.py.
"""

# Libraries
from keras.datasets import mnist
import numpy as np
import random


# Classes
class Network:

    # Methods
    def __init__(self, sizes):
        """
        DESCRIPTION:
        Network constructor.
        :param sizes: [list] ints with the sizes of the layers including 
        the input layer
        """
        self.n_layers = len(sizes)
        self.sizes = sizes
        # Random initialisation of the parameters
        self.biases = [np.random.rand(sizes[i + 1], 1) 
            for i in range(self.n_layers - 1)]
        self.weights = [np.random.rand(sizes[i + 1], sizes[i])
            for i in range(self.n_layers - 1)]
        
    def fit(self, train_x, train_y, epochs=5, batch_size=12):
        """
        DESCRIPTION:
        The function to train the network given the training samples, and its
        labels.
        :param train_x: [np.ndarray] all the samples to train in a 3D tensor.
        :param train_y: [np.ndarray] a nx1 tensor with all the labels for the
        training samples.
        :param epochs: [int] number of epochs to compute during the training.
        :param batch_size: [int] the sample number in every batch-.
        """
        # Divide the data in several batches
        n_batches = int(np.round(train_x.shape[0] / batch_size))
        train_x_batches = np.split(train_x, n_batches)
        train_y_batches = np.split(train_y, n_batches)
        # For every epoch and batch
        for i in range(epochs):
            for j in range(n_batches):
                # Update the parameters
                self.update_parameters(train_x_batches[j], train_y_batches[j])
            # Inform about the accuracy
            print(f'Epoch {i + 1} completed')

    def update_parameters(self, train_x, train_y):
        """
        DESCRIPTION:
        A function to update the parameters given a training sample group.
        :param train_x: [np.ndarray] training samples.
        :param train_y: [np.ndarray] correct labels for the samples in
        train_x.
        """



# Functions
def relu(z):
    """
    DESCRIPTION:
    Activation function ReLU.
    Source: DeepAI.
    :param z: [numpy.ndarray] the vector with all the inputs for the layer.
    :return: [numpy.ndarray] the vector with the outcome of the activation 
    function.
    """
    z[z < 0] = 0
    return z


def softmax(z):
    """
    DESCRIPTION:
    Activation function SoftMax.
    Source: DeepAI.
    :param z: [numpy.ndarray] the vector with all the inputs for the layer.
    :return: [numpy.ndarray] the vector with the outcome of the activation 
    function.
    """
    return np.exp(z) / np.sum(np.exp(z))


def sparse_categorical_crossentropy(truth, predicted):
    """
    DESCRIPTION:
    The loss function sparse_categorical_crossentropy. 
    Source: Leaky ReLU.
    :param truth: [np.ndarray] array with the real labels.
    :param predicted: [np.ndarray] array with the predicted values by the
    model.
    :return: [float] function value.
    """
    return -np.sum(np.log(predicted[range(predicted.shape[0]), truth]))


if __name__ == '__main__':

    # Load data
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # Convert the data to float point numbers
    train_x, test_x  = train_x / 255.0, test_x / 255.0

    # Create the network
    net = Network([784, 128, 10])

    # Train the network
    net.fit(train_x, train_y)

