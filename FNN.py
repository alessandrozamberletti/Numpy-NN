import numpy as np
import matplotlib.pyplot as plt


class FNN:
    def __init__(self, no_input_units, no_hidden_units, no_output_units):
        np.random.seed(1)

        self.l0_weights = self.__random_init((no_input_units, no_hidden_units))
        self.l0_biases = 2 * np.random.random((1, no_hidden_units)) - 1

        self.l1_weights = self.__random_init((no_hidden_units, no_output_units))
        self.l1_biases = 2 * np.random.random((1, no_output_units)) - 1

    @staticmethod
    def __random_init(shape):
        return 2 * np.random.random(shape) - 1

    @staticmethod
    def __transfer(x):
        return 1/(1 + np.exp(-x))

    def __activation(self, x, weights, biases):
        return self.__transfer((np.dot(x, weights) + biases)[0])

    def predict(self, x):
        l0_output = self.__activation(x, self.l0_weights, self.l0_biases)
        l1_output = self.__activation(l0_output, self.l1_weights, self.l1_biases)
        return [int(round(i)) for i in l1_output]

    @staticmethod
    def __plot_loss(mse):
        plt.cla()
        plt.title('Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')

        plt.plot(mse)
        plt.pause(.0001)

    # see: http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
    def fit(self, samples, epochs=1000, lr=.1, momentum=.1, info=True):
        if info:
            plt.ion()

        mse = []
        for epoch in range(epochs):
            square_losses = []
            last_l1_update = 0
            last_l0_update = 0
            for sample in np.random.permutation(samples):
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
                mse.append(np.average(square_losses))
                self.__plot_loss(mse)
                print('Epoch:\t{0}\tMSE:\t{1:.13f}'.format(epoch, np.average(square_losses)))


if __name__ == '__main__':
    data = [([0, 0], [0, 1, 0]),
            ([0, 1], [1, 1, 1]),
            ([1, 1], [0, 0, 1]),
            ([1, 0], [1, 0, 0])]

    input_size = np.shape(data[0][0])[0]
    hidden_size = 2 * np.shape(data[0][0])[0]
    output_size = np.shape(data[0][1])[0]

    fnn = FNN(input_size, hidden_size, output_size)
    fnn.fit(data)
