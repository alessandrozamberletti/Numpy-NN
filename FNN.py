import numpy as np
import matplotlib.pyplot as plt
import random


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


def load_dataset(data_file):
    with open(data_file) as f:
        iris_data = f.read().splitlines()

    class_data_dict = {}
    for line in iris_data:
        data_points = line.split(',')
        features = [float(feat) for feat in data_points[:-1]]
        gt_class = data_points[-1]
        if gt_class not in class_data_dict:
            class_data_dict[gt_class] = []
        class_data_dict[gt_class].append(features)

    classes = {'Iris-setosa': [1, 0],
               'Iris-versicolor': [0, 1],
               'Iris-virginica': [1, 1]}

    train_data = []
    test_data = []
    for gt_class in classes:
        random.shuffle(class_data_dict[gt_class])
        train_size = int(round(len(class_data_dict[gt_class]) * .66))
        train = class_data_dict[gt_class][:train_size]
        test = class_data_dict[gt_class][train_size:]
        [train_data.append((ss, classes[gt_class])) for ss in train]
        [test_data.append((ss, classes[gt_class])) for ss in test]

    return train_data, test_data


if __name__ == '__main__':
    train_set, test_set = load_dataset('iris.data')

    for name, data in [('Train', train_set), ('Test', test_set)]:
        print('{0}\n* samples No.\t{1}\n* features No.\t{2}'.format(name, len(data), len(data[0][0])))

    input_size = np.shape(train_set[0][0])[0]
    hidden_size = 2 * np.shape(train_set[0][0])[0]
    output_size = np.shape(train_set[0][1])[0]

    fnn = FNN(input_size, hidden_size, output_size)
    fnn.fit(train_set)
