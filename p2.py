#!/usr/bin/env python3
import sys, math, csv
import numpy as np

class NeuralNetwork:

    def __init__(self, inputNodes,hiddenNodes,outputNodes ):
        self.iNodes = inputNodes
        self.oNodes = outputNodes
        self.hNodes = hiddenNodes
        self.wInputHidden = np.random.rand(self.hNodes,self.iNodes)
        self.wOutput = np.random.rand(self.oNodes, self.hNodes)
        self.lr = 1.0
        bias = 1.0

    def printStats(self):
        print("number of inputs {}, hidden {}, output {}".format(self.iNodes, self.oNodes, self.hNodes))
        print("wInputHidden")
        print(self.wInputHidden)
        print("wOutput")
        print(self.wOutput)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        ############################# FIRST LAYER
        hidden_inputs = np.dot(self.wInputHidden, inputs)
        hidden_outputs = np.array([self.sigmoid(x) for x in hidden_inputs])
        ############################ SECOND LAYER
        final_inputs = np.dot(self.wOutput, hidden_outputs)
        final_outputs = np.asarray([self.sigmoid(x) for x in final_inputs])
        ################################ ERRORS
        # Calculate ERRORS at the ouput NODES
        output_errors = np.array(targets - final_outputs, ndmin=2)
        ################## BackPropagation STEPS
        outputErrorDerived = output_errors * final_outputs * (1.0 - final_outputs)
        hidden_errors = np.dot(self.wOutput.T, output_errors)
        ################## adjust the weights ########################
        self.wOutput += self.lr * np.dot((outputErrorDerived), np.transpose(hidden_outputs))
        self.wInputHidden += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))



    def query(self, input_list):
        # Create numpy array with input list
        inputs = np.array(input_list).T

        # calculate new hidden inputs from inputs and weights of input layer
        hidden_inputs = (np.dot(self.wInputHidden, inputs))
        # run middle layer through sigmoid
        siged_hidden_outputs = [self.sigmoid(x) for x in hidden_inputs ]
        # multiply outputs with output weights
        final_inputs = np.dot(self.wOutput, siged_hidden_outputs)
        # run last layer through sigmoid
        siged_final_outputs = [self.sigmoid(x) for x in final_inputs]

        return siged_final_outputs

    def sigmoid(self, s):
    # activation function
        return 1/(1+np.exp(-s))


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
                #print(listData)
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

    layer1 = []
    layer2 = []
    for n in neuronList:
        if n.layer == 0:
            layer1.append(n.weightList)
        elif n.layer == 1:
            layer2.append(n.weightList)
    layer1 = np.array(layer1)
    layer2 = np.array(layer2)



    newNN = NeuralNetwork(rowleng,5,3)
    newNN.wInputHidden = layer1
    newNN.wOutput = layer2
    if(internals_flag) == 1:
        newNN.printStats()

########### data normalized
    for y in range(epochs+1):
        epoch_count = 0
        epoch_count_correct = 0
        for x in range(low_trn_rng, hi_trn_rng):
            epoch_count += 1

            csv_row = CSVinputList[x]
            csv_inputs = csv_row[:-1]
            csv_desired = int(csv_row[-1])

            one_hot = [0,0,0]
            one_hot_index = 9

            # init onehot with desired output
            if rowleng < 5:
                one_hot[csv_desired] = 1
                one_hot_index = csv_desired
            else:
                one_hot[csv_desired -1] = 1
                one_hot_index = (csv_desired - 1)
            #print("One hot train")
            #print(one_hot)
            # renamed for repeated use
            inputData = np.array(csv_inputs)
            newNN.train(inputData, one_hot)
            test_nums = newNN.query(inputData)

            index = 0
            max = 0
            for ii in range(len(test_nums)):
                if test_nums[ii] > max:
                    max = test_nums[ii]
                    index = ii
            if index == one_hot_index:
                epoch_count_correct +=1
        if y % 10 == 0:
            if epoch_count_correct != 0:
                percentage = epoch_count_correct / epoch_count
                percentage = round(percentage, 4)
            else:
                percentage = 0
            print("{}   {} correct of {}   {}".format(y,epoch_count_correct, epoch_count, percentage))
            if(internals_flag) == 1:
                newNN.printStats()

    test_count = 0
    test_count_correct = 0
    for t in range(low_tst_rng, hi_tst_rng):
        test_count += 1
        csv_row = CSVinputList[t]
        csv_inputs = csv_row[:-1]
        csv_desired = int(csv_row[-1])
        # print("csv row")
        # print(csv_row)
        one_hot_test = [0,0,0]
        one_hot_index = 9

        # init onehot with desired output
        if rowleng < 5:
            one_hot_test[csv_desired] = 1
            one_hot_index = csv_desired
        else:
            one_hot_test[csv_desired -1] = 1
            one_hot_index = (csv_desired - 1)

        # renamed for repeated use
        inputDataTest = np.array(csv_inputs)
        # print("input data test")
        # print(inputDataTest)
        test_nums = newNN.query(inputDataTest)
        # print("Test Nums")
        # print(test_nums)
        # print("output one hot")
        # print(one_hot_test)
        # print("desired index")
        # print(one_hot_index)
        index_test = 0
        max_test = 0
        for jj in range(len(test_nums)):
            if test_nums[jj] > max:
                max_test = test_nums[jj]
                index_test = jj
        if index_test == one_hot_index:
            test_count_correct += 1
        percentage = test_count_correct / test_count
    print("Test {} correct of {}   {}".format(test_count_correct, test_count, percentage))
    print()



def normalized(smallest, largest, number):
    #print("smallest {} largest {} and number to normalize {}".format(smallest, largest, number))
    normalized = (number - smallest)/(largest - smallest)
    return normalized


if __name__ == '__main__':
    main()
