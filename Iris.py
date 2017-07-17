import numpy as np
import random


class Iris:
    def __init__(self):
        self.classes = {'Iris-setosa': [1, 0],
                        'Iris-versicolor': [0, 1],
                        'Iris-virginica': [1, 1]}

    def __fill_class(self, samples, target):
        class_data = []
        [class_data.append((sample, target)) for sample in samples]
        return class_data

    def __load_classes(self, file_path):
        with open(file_path) as f:
            data = f.read().splitlines()

        class_data_dict = {}
        for line in data:
            data_points = line.split(',')
            features = [float(feat) for feat in data_points[:-1]]
            gt_class = data_points[-1]
            if gt_class not in class_data_dict:
                class_data_dict[gt_class] = []
            class_data_dict[gt_class].append(features)
        return class_data_dict

    def __split(self, iris_data):
        random.seed(1)
        for gt_class in self.classes:
            random.shuffle(iris_data[gt_class])

            train_size = int(round(len(iris_data[gt_class]) * .6))
            test_size = int(np.ceil((len(iris_data[gt_class]) - train_size) / 2))

            train_split = iris_data[gt_class][:train_size]
            test_split = iris_data[gt_class][train_size:train_size + test_size]
            validation_split = iris_data[gt_class][train_size + test_size:]

            train_data = self.__fill_class(train_split, self.classes[gt_class])
            validation_data = self.__fill_class(validation_split, self.classes[gt_class])
            test_data = self.__fill_class(test_split, self.classes[gt_class])
        return train_data, validation_data, test_data

    def load(self, file_path):
        iris_data = self.__load_classes(file_path)
        train, validation, test = self.__split(iris_data)

        for name, data in [('Train', train), ('Validation', validation), ('Test', test)]:
            print('{0}\n* samples No.\t{1}\n* features No.\t{2}'.format(name, len(data), len(data[0][0])))

        return train, validation, test

