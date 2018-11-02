import time
import sys
from random import shuffle

import matplotlib.pyplot as plt

from progressbar import showProgress
from network import Network

training_fraction = 0.7 # Ammount of data that will be used for training
                        # The remaining entries will be used for testing
                        # the precition and recall.

def getResultArray(n, length):
    """Returns an array with zeroes in all indexes
    except on the n-th position."""
    r = [0]*3
    r[n-1] = 1
    return r

def getDataSet(lines, nResults):
    """Converts a list of strings into an array
    of floats and int to use as input/expected_outputs
    for the neural network."""
    result = [[],[]]

    for line in lines:
        splitted = line.split(',')

        for i in range(len(splitted)):
            if i == 0:
                splitted[i] = int(splitted[i])
            else:
                splitted[i] = float(splitted[i])

        wineType = getResultArray(splitted[0], nResults)
        wineProperties = splitted[1:]
        result[0].append(wineProperties)
        result[1].append(wineType)

    return result

def normalize_dataset(original):
    for i in range(len(original[0])):
        aMin = None
        aMax = None

        for j in range(len(original)):
            v = original[j][i]
            if (aMin is None) or (v < aMin):
                aMin = v
            if (aMax is None) or (v > aMax):
                aMax = v

        for j in range(len(original)):
            v = original[j][i]
            original[j][i] = (v - aMin)/(aMax - aMin)

    return original


if __name__ == "__main__":
    result_options = 3 # How many different values the result can have.

    # Open data file
    datafile = open('wine.data', 'r')
    datalines = datafile.readlines()
    shuffle(datalines)    

    # Transform string lists to inputs understandable by the network
    raw_dataset = getDataSet(datalines, 3)
    norm_data = [normalize_dataset(raw_dataset[0]), raw_dataset[1]]

    # Separate training set from testing set
    limit_index = int(len(datalines) * training_fraction)
    training_set = [norm_data[0][0:limit_index], norm_data[1][0:limit_index]]
    testing_set = [norm_data[0][limit_index:], norm_data[1][limit_index:]]

    startt = time.time()
    # Create the network
    n = Network(13, [7,5,5,3])
    nEpochs = int(sys.argv[1])

    # Make the graph
    errores = []
    outputs_length = len(raw_dataset[1][0])
    precitions = []
    recalls = []
    for i in range(outputs_length):
        precitions.append([])
        recalls.append([])

    for i in range(nEpochs):
        error = n.epoch(training_set[0], training_set[1])
        errores.append(error)

        precition = n.epoch_precition(testing_set[0], testing_set[1])
        precitions[0].append(precition[0])
        precitions[1].append(precition[1])
        precitions[2].append(precition[2])

        recall = n.epoch_recall(testing_set[0], testing_set[1])
        recalls[0].append(recall[0])
        recalls[1].append(recall[1])
        recalls[2].append(recall[2])

        showProgress(i, nEpochs)
    print("\nEntrenamiento terminado!")
    endt = time.time()
    print(endt - startt)

    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    f.subplots_adjust(hspace=0.65)
    ax1.set_xlabel("Epochs")
    ax2.set_xlabel("Epochs")
    ax3.set_xlabel("Epochs")

    # Error graph
    ax1.plot(range(nEpochs), errores)
    ax1.set_title('Error absoluto')
    
    # Precition graph
    ax2.plot(range(nEpochs), precitions[0],'b')
    ax2.plot(range(nEpochs), precitions[1],'r')
    ax2.plot(range(nEpochs), precitions[2],'g')
    ax2.set_title('Precision')
    
    # Recall graph
    ax3.plot(range(nEpochs),recalls[0],'b')
    ax3.plot(range(nEpochs),recalls[1],'r')
    ax3.plot(range(nEpochs),recalls[2],'g')
    ax3.set_title('Recall')
    plt.show()
