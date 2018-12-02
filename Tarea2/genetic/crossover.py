# crossover.py
# Algoritmos de cross-over
# @author: Mortup (Gonzalo Uribe)

import random

import selection

# Divide s1 y s2 en dos partes y entrega la union.
def single_point(s1, s2):
    assert len(s1) == len(s2)
    assert len(s1) > 1

    p = random.randint(1, len(s1) - 1)

    return s1[:p] + s2[p:]

# Agrega una parte de s2 a s1.
def two_point(s1, s2):
    assert len(s1) == len(s2)
    assert len(s1) > 2

    p1 = random.randint(1, len(s1) - 2)
    p2 = random.randint(p1, len(s1) - 1)

    return s1[:p1] + s2[p1:p2] + s1[p2:]

# Para cada letra se selecciona la de s1 o s2 con 
# probabilidad uniforme.
def uniform(s1, s2):
    assert len(s1) == len(s2)

    result = []
    for i in range(len(s1)):
        if random.random() > 0.5:
            result.append(s1[i])
        else:
            result.append(s2[i])

    return result

# Genera 'n_childs' hijos con padres seleccionados
# aleatoreamente desde 'pool' utilizando el metodo
# de crossover 'method'.
def generate_childs(n_childs, pool, method):
    result = []
    for i in range(n_childs):
        parent1 = selection.tourneament_selection(pool)
        parent2 = selection.tourneament_selection(pool)
        new_child = method(parent1, parent2)
        result.append(new_child)

    return result
