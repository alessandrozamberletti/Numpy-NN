import numpy as np
import os


class IrisManager:
    def __init__(self, iris_path):
        self.classes = {'Iris-setosa': [1, 0], 'Iris-versicolor': [0, 1], 'Iris-virginica': [1, 1]}
        self.data_dict = {}
        with open(iris_path) as f:
            for line in f.read().splitlines():
                data_points = line.split(',')
                features = [float(feat) for feat in data_points[:-1]]
                gt_class = data_points[-1]
                if gt_class not in self.data_dict:
                    self.data_dict[gt_class] = []
                self.data_dict[gt_class].append(features)

    def __fill(self, data, gt_class, selection):
        return data.extend([(sample, self.classes[gt_class]) for sample in self.data_dict[gt_class][selection]])

    def split(self):
        train_data = []
        validation_data = []
        test_data = []
        for gt_class in self.data_dict:
            train_size = int(round(len(self.data_dict[gt_class]) * .6))
            validation_size = int(np.ceil((len(self.data_dict[gt_class]) - train_size) / 2))

            self.__fill(train_data, gt_class, slice(None, train_size))
            self.__fill(validation_data, gt_class, slice(train_size, train_size + validation_size))
            self.__fill(test_data, gt_class, slice(train_size + validation_size, None))

        return train_data, validation_data, test_data


if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), 'res', 'iris.data')
    train, validation, test = IrisManager(data_path).split()
