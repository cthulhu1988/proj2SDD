#!/usr/bin/env python3
import sys

def main():
    test = "this now works"




# Normalize  numbers between 0 and 1
# Normalize = number - smallest / the difference between the numbers
# so for 1 and 5 -> 1 -1 = 0 / d=4 = 0

# alpha is the momentum and we will not use that
# we will not adjust the biases.
# The Learning rate N will be set to 1 so adding it is irrelevant
# we need to code for two possibilities. First that we are in the hidden layer or layers and second the output nodes.

# def backprop(network, expected):
#     for i in reversed(range(len(network))
#         layer = network[i]
#         error = []
#         if i != len(network) - 1:
#             for j in range(len(layer)):
#                 error = 0.0
#                 for neuron in network[i+1]:
#                     error += 2(neuron[w][j]) * neuron(dolab??)
#                     append(??)
#         else:
#             for j in range(len(layer)):
#                 neuron = layer[j]
#                 error.append(expected[j] - neuron(???))
#                 for j in range(len(layer)):
#                     neuron = jay?[j]
# 
# def expected(network, layer(or row??), learning_rate):
#     for in in range len(layer):
#         input = row
#         if i != 0:
#             input = neuron
#         expected = [0 for i in range(num of rates??)]




# derivative function
def sigmoidPrime(x):
    return (sigmoid(x) - (1-sigmoid(x)))


# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))





if __name__ == '__main__':
    main()
