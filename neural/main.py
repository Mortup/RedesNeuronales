import matplotlib.pyplot as plt

from progressbar import showProgress
from network import Network

def getResultArray(n):
    r = [0,0,0]
    r[n-1] = 1
    return r

def getDataSet():
    f = open('wine.data', 'r')
    lines = f.readlines()

    result = [[],[]]

    for line in lines:
        splitted = line.split(',')

        for i in range(len(splitted)):
            if i == 0:
                splitted[i] = int(splitted[i])
            else:
                splitted[i] = float(splitted[i])

        wineType = getResultArray(splitted[0])
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
    raw_dataset = getDataSet()
    training_set = [normalize_dataset(raw_dataset[0]), raw_dataset[1]]

    n = Network(13, [7,5,5,3])
    nEpochs = 100000

    errores = []

    for i in xrange(nEpochs):
        error = n.epoch(training_set[0], training_set[1])
        errores.append(error)
        showProgress(i, nEpochs)
    print("\nEntrenamiento terminado!")

    print(n.feed(normalize_dataset(raw_dataset[0])[2]))
    print(n.feed(normalize_dataset(raw_dataset[0])[103]))
    print(n.feed(normalize_dataset(raw_dataset[0])[161]))

    plt.plot(range(nEpochs), errores)
    plt.show()
