
# Libraries
from keras.datasets import mnist
from matplotlib import pyplot as plt
import matplotlib
import random
import numpy as np

# Classes
class Network:
    # Methods
    def __init__(self, sizes):
        """
        DESCRIPTION: class constructor.
        """
        # Store the network information
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Random parameter initialisation
        self.biases = [np.random.rand(size, 1) for size in sizes[1:]]
        self.weights = [np.random.rand(y, x) for (x, y) in zip(sizes[:-1], sizes[1:])]
        self.biases[0] = 0.001 * self.biases[0]
        self.weights[0] = 0.001 * self.weights[0]

    def feedforward(self, a):
        """
        DESCRIPTION: a method to compute the network from the initial to the
        last layers.
        Equation: a' = wa + b
        :param a: [np.ndarray] n x 1 input vector.
        :return: [float] network outcome
        """
        a = np.reshape(a, (a.shape[0] ** 2, 1))
        for (b, w) in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def update_mini_batch(self, mini_batch, eta):
        """
        DESCRIPTION:
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        :param mini_batch: [list] tuples (x, y), image to classify (numpy.array, x) and
        correct classification (int, y).
        :param eta: [float] learning rate.
        """
        # Store space in memory for the parameter gradients
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Update the parameters for every example in the mini batch
        for x, y in mini_batch:
            # Compute the gradients
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # Sum the gradients for every batch
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        """
        DESCRIPTION:
        Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x.  "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.biases" and "self.weights".
        :param x: [np.array] the image to classify.
        :param y: [int] the correct class of the image.
        :return: [tuple] np.arrays with the gradient vectors of both 
        biases and weights.
        """
        # My solution
        # Feedforward
        # Obtain the input vector z for the last layer
        activations = [np.reshape(x, (x.shape[0] ** 2, 1))]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activations.append(sigmoid(z))
        # Compute delta (error) for that layer
        y_digit = y - 1
        y = np.zeros(activations[-1].shape)
        y[y_digit] = 1
        nabla_c = activations[-1] - y
        # Backpropagate the error in the other layers
        n_layers = len(self.biases)
        deltas = [0.0] * n_layers
        deltas[-1] = nabla_c * sigmoid_prime(zs[-1])
        for i in range(n_layers - 2, -1, -1):
            deltas[i] = np.dot(self.weights[i + 1].transpose(), deltas[i + 1]) * sigmoid_prime(zs[i])
        # Build the parameter update
        nabla_b = deltas
        nabla_w = [np.dot(deltas[i], activations[i].transpose()) for i in range(len(deltas))]
        return (nabla_b, nabla_w)
        """
        # Store information for the computing
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Feedforward
        x = np.reshape(x, (x.shape[0] ** 2, 1))
        activation = x
        activations = [x] # List to store all the activations, layer by layer
        zs = [] # List to store all the z vectors, layer by layer
        # Iterate over all the parameters for all the neurons
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
        """

    def evaluate(self, test_data):
        """
        DESCRIPTION:
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        :param test_data: [list] the numpy arrays representing the image list 
        to test the algorithm while training.
        :return: [int] number of images correctly classified.
        """
        n_corrects = sum([int((np.argmax(self.feedforward(image)) - 1) == correct_classification) 
            for image, correct_classification in test_data])
        return n_corrects

        return int(n_correct)            
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        """
        
    def cost_derivative(self, output_activations, y):
        """
        DESCRIPTION:
        Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.
        """
        return (output_activations-y)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        DESCRIPTION: stochastic gradient descent algorithm. Train the neural
        network using mini-batch stochastic gradient descent.  The 
        "training_data" is a list of tuples "(x, y)" representing the training
        inputs and the desired outputs. Essentially, the images are divided in 
        batches and then the parameters are updated for every batch.
        :param training_data: [list] the numpy arrays representing the image list 
        to train the algorithm.
        :param test_data: [list] the numpy arrays representing the image list 
        to test the algorithm while training.
        :param epochs: [int] number of times that the training_data should be 
        computed.
        :param mini_batch_size: [int] number of images to compute the gradient.
        :param eta: [float] the learning rate.
        """
        # Go over the data in all the epochs
        for i in range(epochs):
            # Shuffle the data to take different combination in every epoch
            random.shuffle(training_data)
            # Create the mini batches to compute the gradient
            mini_batches = [training_data[k:k+mini_batch_size] 
            for k in range(0, len(training_data), mini_batch_size)]
            # Update the parameters for every mini batch
            [self.update_mini_batch(mini_batch, eta) for mini_batch in mini_batches]
            # Compare with test data
            if test_data:
                # Return the number of test examples from which the network returns
                # the correct output
                print(f'Epoch {i + 1}: {self.evaluate(test_data)} / {len(test_data)} correct')
            else:
                print(f'Epoch {i + 1} completed')
                

# Functions
def sigmoid(z):
    """
    DESCRIPTION:
    Sigmoid function.
    """
    result = 1.0 / (1.0 + np.exp(-z))
    return result

def sigmoid_prime(z):
    """
    DESCRIPTION:
    Derivative of the sigmoid function.
    """
    return sigmoid(z)*(1-sigmoid(z))


if __name__ == '__main__':

    # Load data
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # # Convert to float numbers
    # train_x = train_x / 255.0
    # test_x = test_x / 255.0

    # Convert to image list
    train_x = np.split(train_x, train_x.shape[0], axis=0)
    train_x = list(map(lambda image: np.squeeze(image).astype('float128'), train_x))
    training_data = list(zip(train_x, train_y))
    test_x = np.split(test_x, test_x.shape[0], axis=0)
    test_x = list(map(lambda image: np.squeeze(image).astype('float128'), test_x))
    test_data = list(zip(test_x, test_y))

    # Create the network
    net = Network([784, 30, 10])

    # Parameters to train
    training_data = training_data
    epochs = 30
    mini_batch_size = 10
    eta = 3.0
    test_data = test_data

    # Train the network
    net.SGD(training_data, epochs, mini_batch_size, eta, test_data=test_data)




