# selection.pt
# Algoritmos de seleccion de los elementos con
# mayor fitness.
# @author: Mortup (Gonzalo Uribe)

# Devuelve los 'n_elements' elementos con mayor fitness.
def top_elements(pool, n_elements, fitness_func):
    assert len(pool) >= n_elements
    fitnesses = []

    for i in range(len(pool)):
        fitness = fitness_func(pool[i])
        fitnesses.append(fitness)

    childs_with_fitness = zip(pool, fitnesses)
    sorted_childs = sorted(childs_with_fitness, key=lambda x: x[1], reverse=True)
    
    top_elements = sorted_childs[:n_elements]
    return [x[0] for x in top_elements]
