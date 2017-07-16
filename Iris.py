import numpy as np
import random


def populate_class(samples, target):
    class_data = []
    [class_data.append((sample, target)) for sample in samples]
    return class_data


def load_iris(data_file):
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

    random.seed(1)
    for gt_class in classes:
        random.shuffle(class_data_dict[gt_class])
        train_size = int(round(len(class_data_dict[gt_class]) * .6))
        test_size = int(np.ceil((len(class_data_dict[gt_class]) - train_size)/2))
        train_split = class_data_dict[gt_class][:train_size]
        test_split = class_data_dict[gt_class][train_size:train_size+test_size]
        validation_split = class_data_dict[gt_class][train_size+test_size:]
        train_data = populate_class(train_split, classes[gt_class])
        test_data = populate_class(test_split, classes[gt_class])
        validation_data = populate_class(validation_split, classes[gt_class])

    for name, data in [('Train', train_data), ('Test', test_data), ('Validation', validation_data)]:
        print('{0}\n* samples No.\t{1}\n* features No.\t{2}'.format(name, len(data), len(data[0][0])))

    return train_data, test_data, validation_data
