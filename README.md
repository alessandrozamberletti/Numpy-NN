# Numpy-NN
NumPy Feedforward Neural Network

![neural_network-training](https://github.com/alessandrozamberletti/NumPy-NN/blob/master/res/numpy-nn.gif)

```python
train, validation, test = IrisManager('res/iris.data').split()
neural_network = NeuralNetwork(4, 8, 2)
neural_network.fit(train, validation)
```
