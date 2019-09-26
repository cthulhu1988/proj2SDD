#!/usr/bin/env python3.7
#####  use python3.7 on system64 <--------------------------------------------------------

# CLASSIFY PROBLEM:  categories are 1, 2, 3 (as onehots)


import sys, random, time, os

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.metrics import binary_crossentropy, categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping


stime = time.time()

# with open("p2testfiles/SEEDS.csv") as infile:
with open("p2testfiles/IRIS.csv") as infile:
    header = infile.readline()
    data = []
    for line in infile:
        sline = line.strip().split(',')
        sline[-1] = int(sline[-1])
        for idx in range(len(sline)-1):
            sline[idx] = float(sline[idx])
        data.append(sline)
    random.shuffle(data)
    labels = sorted( set( [sline[-1] for sline in data] ) )
    # create onehots for labels
    onehots = []
    for sline in data:
        onehot = [0] * len(labels)
        onehot[labels.index(sline[-1])] = 1
        onehots.append(onehot)
    labels = onehots
    # normalize data
    numInputs = len(data[0]) - 1  # all but labels
    print("DBG",data[0])
    for colidx in range( numInputs ):
        tot = 0.0
        maxval = 0.0
        minval = 999999.0
        for row in data:
            colval = row[colidx]
            if colval < minval:
                minval = colval
            if colval > maxval:
                maxval = colval
        diffval = maxval - minval
        for row in data:
            row[colidx] = (row[colidx] - minval) / diffval
    print("DBG",data[0])


validation_split = 0.2
lentrain = int( len(data) * (1.0-validation_split) )
print(lentrain)
lentest = len(data) - lentrain

trainX = np.array(data[:lentrain], dtype=float)
testX  = np.array(data[lentrain:], dtype=float)
trainy = np.array(labels[:lentrain], dtype=float)
testy  = np.array(labels[lentrain:], dtype=float)
print("DBG",trainX.shape,testX.shape,trainy.shape,testy.shape)

checkpoint = ModelCheckpoint("catMODEL.h5", monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')
earlystop  = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
callbacks_list = [checkpoint,earlystop]

(verbose,batch_size,epochs) = (2,10,50)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(trainX.shape[1],)))
model.add(Dropout(0.5))
## model.add(Dense(64, activation='relu'))
## model.add(Dropout(0.5))
## model.add(Dense(16, activation='relu'))    # 93% with 11 epochs
## model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("BEGIN FIT %.2f" %(time.time()-stime) )

model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size,
          validation_data=(testX,testy),
          callbacks=callbacks_list, verbose=verbose)

print("END FIT %.2f" %(time.time()-stime) )

_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

print("ACC", accuracy)

exit(0)
