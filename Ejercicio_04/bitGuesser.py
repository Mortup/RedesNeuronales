import random

def generateRandomBits(n):
    result = ''
    for i in range(n):
        b = str(random.randint(0,1))
        result += b

    return result

def generateStartingPopulation(n, elementSize):
    result = []

    for i in range(n):
        result.append(generateRandomBits(elementSize))

    return result

def getFitness(obtained, desired):
    assert len(obtained) == len(desired)

    fitness = 0
    for i in range(len(obtained)):
        if obtained[i] == desired[i]:
            fitness += 1

    return fitness

def selectTopPercent(results, desired, percent):
    # Evaluate fitness of each element
    fitnesses = []
    for i in range(len(results)):
        fitnesses.append(getFitness(results[i], desired))

    res_and_fit = zip(results, fitnesses)
    res_and_fit = sorted(res_and_fit, key=lambda x: x[1], reverse=True)
    top_res_and_fit = res_and_fit[:len(res_and_fit)*percent/100]
    return top_res_and_fit

def getCrossedOver(results)

a = generateStartingPopulation(20, 5)
b = selectTopPercent(a, '11111', 25)
print(b)

