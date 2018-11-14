import genetic.generator as generator
import genetic.selection as selection
import genetic.crossover as crossover
import genetic.mutation as mutation

def fitness_function_factory(desired):
    return lambda x: getFitness(x, desired)

def getFitness(obtained, desired):
    assert len(obtained) == len(desired)

    fitness = 0
    for i in range(len(obtained)):
        if obtained[i] == desired[i]:
            fitness += 1

    return fitness

n = raw_input("Ingrese una palabra: ")

alphabet = 'qwertyuiopasdfghjklzxcvbnm '
child_length = len(n)
population_size = 20
top_percent_size = 5
mutation_rate = 0.1
fitness_func = fitness_function_factory(n)

population = generator.generateRandomSet(child_length, alphabet, population_size)
top_percent = selection.top_elements(population, top_percent_size, fitness_func)

iterations = 0
while top_percent[0] != n:
    c_pool = crossover.generate_childs(population_size, top_percent, crossover.uniform)
    m_pool = mutation.mutate_pool(c_pool,alphabet,mutation_rate) 
    top_percent = selection.top_elements(m_pool, top_percent_size, fitness_func)
    iterations += 1

print "Se ha encontrado la palabra", n, "luego de", iterations, "generaciones."
