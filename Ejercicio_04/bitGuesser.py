import random

import selection
import crossover
import mutation

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

def guessNumber(n):
    child_length = len(n)
    population_size = 20
    top_percent_size = 25
    mutation_rate = 0.1

    population = generateStartingPopulation(population_size, child_length)
    top_percent = selectTopPercent(population, n, top_percent_size)
    iterations = 0
    while top_percent[0][0] != n:
        pool = [x[0] for x in top_percent]
        c_pool = crossover.generate_childs(5, pool, crossover.uniform)
        m_pool = mutation.mutate_pool(c_pool,'01',mutation_rate) 
        top_percent = selectTopPercent(m_pool, n, top_percent_size)
        iterations += 1

    print top_percent[0]
    print iterations

numero = raw_input("Numero: ")
guessNumber(numero)
