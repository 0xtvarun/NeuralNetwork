import numpy as np
from activation import sigmoid, sigmoid_prime
from sklearn.utils import shuffle

class Network(object):

    def __init__(self, *args, **kwargs):
        # hyper parameters
        self.size = kwargs.pop('size', None)

        self.weights = kwargs.pop('weights', [np.random.randn(y, x) for x, y in zip(self.size[:-1], self.size[1:])])
        self.biases = kwargs.pop('biases', [np.random.randn(y, 1) for y in self.size[1:]])

    def feed_forward(self, a):
        for weight, biase in zip(self.weights, self.biases):
            a = sigmoid(np.dot(weight, a) + biase)
        return a

    def backprop(self, x, y):
        new_biase = [np.zeros(b.shape) for b in self.biases]
        new_weight = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]  # list to store all the activations, layer by layer

        zs = []  # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        new_biase[-1] = delta
        new_weight[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            new_biase[-l] = delta
            new_weight[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (new_biase, new_weight)

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        n_test = 10000
        n = 60000
        for j in range(epochs):
            shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print
                "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print
                "Epoch {0} complete".format(j)

        with open('network.pickle') as p:
            p.write(self.weights)
            p.write(self.biases)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        print(mini_batch)
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
