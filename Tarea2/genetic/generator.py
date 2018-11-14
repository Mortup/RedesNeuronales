# generator.py
# Modulo que genera palabras al azar
# @author: Mortup (Gonzalo Uribe)

import random

# Genera una palabra de largo 'n' con letras
# de 'alphabet' seleccionadas al azar.
def generateRandomWord(n, alphabet):
    result = ''

    for i in range(n):
        character = random.choice(alphabet)
        result += character

    return result

# Genera un set de 'n' elementos de largo 'l'
# con letras del alfabeto 'alphabet'.
def generateRandomSet(l, alphabet, n):
    result = []

    for i in range(n):
        word = generateRandomWord(l, alphabet)
        result.append(word)

    return result
