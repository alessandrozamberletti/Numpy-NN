import numpy as np
import matplotlib.pyplot as plt

from IrisManager import IrisManager


class NeuralNetwork:
    def __init__(self, no_input, no_hidden, no_output):
        np.random.seed(1)

        self.l0_weights = self.__random_init((no_input, no_hidden))
        self.l0_biases = 2 * np.random.random((1, no_hidden)) - 1

        self.l1_weights = self.__random_init((no_hidden, no_output))
        self.l1_biases = 2 * np.random.random((1, no_output)) - 1

    @staticmethod
    def __random_init(shape):
        return 2 * np.random.random(shape) - 1

    @staticmethod
    def __transfer(x):
        return 1/(1 + np.exp(-x))

    def __activation(self, x, weights, biases):
        return self.__transfer((np.dot(x, weights) + biases)[0])

    @staticmethod
    def __plot_loss(mse, test_mse):
        plt.cla()
        plt.title('Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')

        plt.plot(mse, label='train')
        plt.plot(test_mse, label='validation')
        plt.legend()
        plt.pause(.0001)

    def evaluate(self, data):
        l0_output = self.__transfer((np.dot([el[0] for el in data], self.l0_weights) + self.l0_biases))
        l1_output = self.__transfer((np.dot(l0_output, self.l1_weights) + self.l1_biases))

        return np.average(np.power([el[1] for el in data] - l1_output, 2))

    # see: http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
    def fit(self, train_samples, test_samples, epochs=100, lr=.1, momentum=.1, info=True):
        if info:
            plt.ion()

        train_mse = []
        validation_mse = []
        for epoch in range(epochs):
            square_losses = []
            last_l1_update = 0
            last_l0_update = 0
            for sample in np.random.permutation(train_samples):
                observation, expectation = sample
                # move forward
                l0_output = self.__activation(observation, self.l0_weights, self.l0_biases)
                l1_output = self.__activation(l0_output, self.l1_weights, self.l1_biases)
                # move backward
                loss = expectation - l1_output
                square_losses.append(loss**2)
                l1_error = loss * (l1_output * (1 - l1_output))
                l0_error = np.dot(self.l1_weights, l1_error) * (l0_output * (1 - l0_output))
                # compute weight updates
                last_l1_update = l1_update = momentum * last_l1_update + lr * np.outer(l0_output, l1_error)
                last_l0_update = l0_update = momentum * last_l0_update + lr * np.outer(observation, l0_error)
                # update weights and biases
                self.l0_weights += l0_update
                self.l0_biases += l0_error
                self.l1_weights += l1_update
                self.l1_biases += l1_error

            if info:
                train_mse.append(np.average(square_losses))
                validation_mse.append(self.evaluate(test_samples))
                self.__plot_loss(train_mse, validation_mse)
                print('Epoch:\t{0}\t'
                      'Train MSE:\t{1:.13f}\t'
                      'Validation MSE:\t{2:.13f}'.format(epoch, np.average(square_losses), validation_mse[-1]))

if __name__ == '__main__':
    train, validation, test = IrisManager('res/iris.data').split()
    neural_network = NeuralNetwork(4, 8, 2)
    neural_network.fit(train, validation)
