from progressbar import showProgress
from network import Network

def getResultArray(n):
    r = [0,0,0]
    r[n-1] = 1
    return r

def getDataSet():
    f = open('wine.data', 'r')
    lines = f.readlines()

    result = []

    for line in lines:
        splitted = line.split(',')
        wineType = getResultArray(splitted[0])
        wineProperties = splitted[1:]
        result.append([wineProperties, wineType])

    for a in result:
        print a
    return result


if __name__ == "__main__":
    getDataSet()
    n = Network(2, [2,3,3,2])
    nEpochs = 100000

    for i in xrange(nEpochs):
        n.epoch([[0,0],[1,0],[0,1],[1,1]],[[0,1],[1,0],[1,0],[0,1]])
        showProgress(i,nEpochs,"Training")
    print("\nEntrenamiento terminado!")
