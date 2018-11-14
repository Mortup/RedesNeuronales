# mutation.py
# Contiene algoritmos de mutacion de strings para
# algoritmos geneticos.
# @author: Mortup (Gonzalo Uribe)

import random

# Reemplaza una letra de una palabra por otra
# seleccionada al azar de 'alphabet'
def mutate(gen, alphabet):
    index = random.randint(0, len(gen) - 1)
    new_symbol = random.choice(alphabet)

    s = list(gen)
    s[index] = new_symbol
    return "".join(s)
    
# Aplica mutaciones en cada palabra de 'pool' con
# probabilidad 'mutation_rate'
def mutate_pool(pool, alphabet, mutation_rate):
    assert 0 <= mutation_rate <= 1

    result = []
    for gen in pool:
        if random.random() < mutation_rate:
            mutated_gen = mutate(gen, alphabet)
            result.append(mutated_gen)
        else:
            result.append(gen)

    return result
