# Algoritmos de cross-over

import random

def single_point(s1, s2):
    assert len(s1) == len(s2)
    assert len(s1) > 1

    p = random.randint(1, len(s1) - 1)

    return s1[:p] + s2[p:]

def two_point(s1, s2):
    assert len(s1) == len(s2)
    assert len(s1) > 2

    p1 = random.randint(1, len(s1) - 2)
    p2 = random.randint(p1, len(s1) - 1)

    return s1[:p1] + s2[p1:p2] + s1[p2:]

def uniform(s1, s2):
    assert len(s1) == len(s2)

    result = ""
    for i in range(len(s1)):
        if random.random() > 0.5:
            result += s1[i]
        else:
            result += s2[i]

    return result

def generate_childs(n_childs, pool, method):
    result = []
    for i in range(n_childs):
        parent1 = random.choice(pool)
        parent2 = random.choice(pool)
        new_child = method(parent1, parent2)
        result.append(new_child)

    return result
