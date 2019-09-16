#!/usr/bin/env python3
import sys
import math

class Neuron:
    def __init__(self, layer, neuron, weightList, bias):
        self.layer = layer
        self.neuron = neuron
        self.weightList = weightList
        self.bias = bias
    def printStats(self):
        print("layer {} neuron {} weightList {} and BIAS: {}".format(self.layer, self.neuron, self.weightList, self.bias))

def NeuronMaker(listData):
    layer = int(listData[0])
    neuron = int(listData[1])
    bias = float(listData[-1])
    weightData = listData[2:-1]
    weightData = [float(i) for i in weightData]
    return Neuron(layer, neuron, weightData, bias)

def main():

    inputData = []
    layer = 0
    neuron = 0
    bias = 0
    matrixList =[]

    newFile = sys.argv[1]
    with open(newFile, 'r') as fp:
        line = fp.readline()
        inputs=''
        while line:
            # line with comments
            parsePound = line.find('#')
            if parsePound >= 0 and line[0] != 'i':
                data = line[0:parsePound]
            #input line, 2 or more values
            elif line[0] == 'i':
                parsePound = line.find('#')
                inputs = line[8:parsePound]
                inputs = inputs.strip()
                inputData = inputs.split()
                inputData = [float(i) for i in inputData]
            # regular lines
            else:
                data = line
            data = data.strip()
            listData = data.split()
            if len(listData)>0:
                matrixList.append(NeuronMaker(listData))


            line = fp.readline()
    fp.close()

    currentLayer = 0
    newHiddenList=[]
    sum = 0
    for n in matrixList:
        if n.layer == currentLayer:
            for x in range(len(inputData)):
                sum += n.weightList[x] * inputData[x]
            sum += n.bias
            newHiddenList.append(sigmoid(sum))
            sum = 0
            print()
        else:
            inputData = newHiddenList
            newHiddenList = []
            sum = 0
            currentLayer +=1
            for x in range(len(inputData)):
                sum += n.weightList[x] * inputData[x]
            sum+=n.bias
            newHiddenList.append(sigmoid(sum))
            sum = 0
    print("Final output: ", newHiddenList)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))



if __name__ == '__main__':
    main()
