import matplotlib.pyplot as plt
import numpy as np


# Class Definition
class NeuralNetwork(object):

    def __init__(self, num_params=2):
        self.weight_matrix = 2 * np.random.random((num_params + 1, 1)) - 1
        self.l_rate = 1

    def hard_limiter(self, x):
        outs = np.zeros(x.shape)
        outs[x > 0] = 1
        return outs

    def forward_propagation(self, inputs):
        outs = np.dot(inputs, self.weight_matrix)
        return self.hard_limiter(outs)

    def train(self, train_inputs, train_outputs, num_train_iterations=10):
        count = 1
        for iteration in range(num_train_iterations):
            for i in range(train_inputs.shape[0]):
                pred_i = self.pred(train_inputs[i, :])
                if pred_i != train_outputs[i]:
                    output = self.forward_propagation(train_inputs[i, :])
                    error = train_outputs[i] - output
                    adjustment = self.l_rate * error * train_inputs[i]
                    self.weight_matrix[:, 0] += adjustment
                    print('Iteration #', count)
                    count += 1
                    plot_fun_thr(train_inputs[:, 1:3], train_outputs, self.weight_matrix[:, 0], classes)

    def pred(self, inputs):
        preds = self.forward_propagation(inputs)
        return preds


# Function Definitions
def plot_fun(features, labels, classes):
    plt.plot(features[labels[:] == classes[0], 0], features[labels[:] == classes[0], 1], 'rs',
             features[labels[:] == classes[1], 0], features[labels[:] == classes[1], 1], 'g^')
    plt.axis([-2, 2, -2, 2])
    plt.xlabel('x: feature 1')
    plt.ylabel('y: feature 2')
    plt.legend(['Class' + str(classes[0]), 'Class' + str(classes[1])])
    plt.show()


def plot_fun_thr(features, labels, thre_parms, classes):
    plt.plot(features[labels[:] == classes[0], 0], features[labels[:] == classes[0], 1], 'rs',
             features[labels[:] == classes[1], 0], features[labels[:] == classes[1], 1], 'g^')
    plt.axis([-2, 2, -2, 2])
    x1 = np.linspace(-2, 2, 50)
    x2 = -(thre_parms[1] * x1 + thre_parms[0]) / thre_parms[2]
    plt.plot(x1, x2, '-r')
    plt.xlabel('x: feature 1')
    plt.ylabel('y: feature 2')
    plt.legend(['Class' + str(classes[0]), 'Class' + str(classes[1])])
    plt.show()


# Main
features = np.array([[1, 1], [1, 0], [0, 1], [-1, -1], [-1, 0], [-1, 1]])
print('Inputs')
print(features)
labels = np.array([1, 1, 0, 0, 0, 0])
desired = np.array([[1], [1], [0], [0], [0], [0]])
print('Desired Labels')
print(desired)
classes = [0, 1]
print('Data Points Plotted')
plot_fun(features, labels, classes)
# Adding Bias
bias = np.ones((features.shape[0], 1))
print('Bias added to Input Array')
features = np.append(bias, features, axis=1)
print(features)
print('Shape of Bias/Input Array Combined')
print(features.shape)
# Training
neural_network = NeuralNetwork(2)
print('\n======Starting Training======')
print('Random weights at the start of training')
print(neural_network.weight_matrix)

neural_network.train(features, labels, 10)
print('New weights after training')
print(neural_network.weight_matrix)
print('=============================')
# Testing Network
print('Testing network on training data points ->')
print(neural_network.pred(features))
# Testing Network on New Samples
print('Testing network on new examples ->')
print('New Test Data Inputs')
test = np.array([[2, 0], [2, 1], [0, 0], [-2, 0]])
print(test)
print('Predicted Labels')
print(neural_network.pred(np.array([2, 0, 1])))
print(neural_network.pred(np.array([2, 1, 1])))
print(neural_network.pred(np.array([0, 0, 1])))
print(neural_network.pred(np.array([-2, 0, 1])))
