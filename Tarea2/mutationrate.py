import genetic.generator as generator
import genetic.selection as selection
import genetic.crossover as crossover
import genetic.mutation as mutation

import matplotlib.pyplot as plt
import random

random.seed(42) # So long and thanks for all the fish

def fitness_func(obtained):
    fitness = 0
    for i in range(len(obtained)):
        for j in range(len(obtained)):
            if j == i:
                continue

            a = obtained[i]
            b = obtained[j]

            if (a[0] == b[0]):
                fitness -= 1
            if (a[1] == b[1]):
                fitness -= 1
            if abs(a[0]-b[0]) == abs(a[1]-b[1]):
                fitness -= 1

    return fitness

# Imprime una combinacion de posiciones de reina
# de manera bonita.
def print_board(positions):
    print "\nImprimiendo tablero:\nx:Casilla vacia\no:Reina\n"
    n = len(positions)

    for y in range(n):
        s = ""
        for x in range(n):
            empty = True
            for pos in positions:
                if pos[0] == x and pos[1] == y:
                    empty = False

            if empty:
                s += "x "
            else:
                s += "o "
        print s



n = int(raw_input("Largo del tablero: "))

alphabet = []
generations_taken = []

for i in range(n):
    for j in range(n):
        alphabet.append((i,j))

child_length = n
population_size = 20
top_percent_size = 5

n_tests = 50
for i in range(n_tests):
    random.seed(42)
    mutation_rate = (float(i)+1)/(n_tests+1)
    
    population = generator.generateRandomSet(child_length, alphabet, population_size)
    top_percent = selection.top_elements(population, top_percent_size, fitness_func)
    
    iterations = 0
    while fitness_func(top_percent[0]) < 0:
        c_pool = crossover.generate_childs(population_size, top_percent, crossover.uniform)
        m_pool = mutation.mutate_pool(c_pool,alphabet,mutation_rate) 
        top_percent = selection.top_elements(m_pool, top_percent_size, fitness_func)
        iterations += 1

    print "Se ha encontrado la combinacion luego de", iterations, "generaciones."
    generations_taken.append(iterations)

f, ax = plt.subplots()
ejex = [1.0/ len(generations_taken) + float(x) / len(generations_taken) for x in range(len(generations_taken))]
ax.plot(ejex, generations_taken)
ax.set_xlabel('Mutation Rate')
ax.set_ylabel('Generaciones')
ax.set_title('Aprendizaje al variar mutation rate')
plt.show()
