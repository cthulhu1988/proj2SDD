#!/usr/bin/env python3
import sys, math, csv

def main():
    test = "this now works"

class Neuron:
    def __init__(self, layer, neuron, weightList, bias):
        self.layer = layer
        self.neuron = neuron
        self.weightList = weightList
        self.bias = bias
        self.value = 0
        self.delta = 0

    def printStats(self):
        print("layer {} neuron {} weightList {} and BIAS {} and value {}".format(self.layer, self.neuron, self.weightList, self.bias, self.value))

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
    neuronList =[]
    # files
    fileNodes = sys.argv[1]
    fileCSV = sys.argv[2]
    # low and high training range
    low_trn_rng = int(sys.argv[3])
    hi_trn_rng = int(sys.argv[4])
    # low and high testing range
    low_tst_rng = int(sys.argv[5])
    hi_tst_rng = int(sys.argv[6])

    epochs = int(sys.argv[7])
    # whether to print extra data
    internals_flag = int(sys.argv[8])


    # OPEN and PARSE NODES file ################################
    with open(fileNodes, 'r') as fp:
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
                neuronList.append(NeuronMaker(listData))


            line = fp.readline()
    fp.close()

    # OPEN and PARSE CSV file ##########################
    with open(fileCSV, newline = '') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        # read whole file.
        CSVinputList = []
        rowleng = 0
        for row in reader:
            rowleng = len(row) -1
            CSVinputList.append(row)

############## Normalize data in list ###################
        count = 0
        min = 1000
        max = 0
        while count < rowleng:
            for item in CSVinputList:
                item[count] = float(item[count])
                if item[count] < min:
                    min = item[count]
                if item[count] > max:
                    max = item[count]
            for floatItem in CSVinputList:
                floatItem[count] = normalized(min, max, floatItem[count])
                #print("normalized {}".format(floatItem[count]))
            min = 1000
            max = 0
            count +=1



########### data normalized

    for x in range(low_trn_rng, hi_trn_rng):
        csv_row = CSVinputList[x]
        csv_inputs = csv_row[:-1]
        csv_desired = int(csv_row[-1])
        one_hot = [0,0,0]

        # init onehot with desired output
        one_hot[csv_desired] = 1
        # renamed for repeated use 
        inputData = csv_inputs

        currentLayer = 0
        newHiddenList=[]
        sum = 0
        for n in neuronList:
            if n.layer == currentLayer:
                for x in range(len(inputData)):
                    sum += n.weightList[x] * inputData[x]
                sum += n.bias
                neuronalSum = sigmoid(sum)
                n.value = neuronalSum
                newHiddenList.append(neuronalSum)
                sum = 0
            else:
                inputData = newHiddenList
                newHiddenList = []
                sum = 0
                currentLayer +=1
                for x in range(len(inputData)):
                    sum += n.weightList[x] * inputData[x]
                sum += n.bias
                neuronalSum = sigmoid(sum)
                n.value = neuronalSum
                newHiddenList.append(neuronalSum)
                sum = 0

        outPutLayerNumber = currentLayer
        #print("Final output newHiddenList: ", newHiddenList)
        #neuronList = reversed(neuronList)
        outputNeuronList = []
        hiddenLayerList = []
        # split up output and hidden
        for m in neuronList:
            if m.layer == outPutLayerNumber:
                outputNeuronList.append(m)
            else:
                hiddenLayerList.append(m)




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


def normalized(smallest, largest, number):
    #print("smallest {} largest {} and number to normalize {}".format(smallest, largest, number))
    normalized = (number - smallest)/(largest - smallest)
    return normalized

# derivative function
def sigmoidPrime(x):
    return (sigmoid(x) - (1-sigmoid(x)))


# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))





if __name__ == '__main__':
    main()
