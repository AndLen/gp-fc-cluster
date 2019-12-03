import operator
from deap import gp, creator
import numpy as np
import random

from deap import tools


def process_data(individual, toolbox, data):
    func = toolbox.compile(individual)
    X = []

    for instance in data:
        # convert each feature to a single element vector for vector representation
        vec = list(map(lambda x: [x], instance))
        X.append(func(*vec))

    return X


def init_primitives(pset):
    pset.addPrimitive(add, 2)
    pset.addPrimitive(add_abs, 2)
    pset.addPrimitive(subtract, 2)
    pset.addPrimitive(subtract_abs, 2)
    pset.addPrimitive(multiply, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(concat, 2)
    pset.addPrimitive(min_v, 2)
    pset.addPrimitive(max_v, 2)
    pset.addPrimitive(if_v, 3)
    pset.addEphemeralConstant("rand", ephemeral=lambda: [random.uniform(-1, 1)])


def init_toolbox(toolbox, pset):

    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=2, max_=8)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=9))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=9))


def fix_vectors(a, b):
    if type(a) == float:
        a = [a]
    if type(b) == float:
        b = [b]
    shortest = min(len(a), len(b))
    a = a[:shortest]
    b = b[:shortest]
    return a, b


def add(a, b):
    a, b = fix_vectors(a, b)
    return np.add(a, b)


def add_abs(a, b):
    return np.absolute(add(a, b))


def subtract(a, b):
    a, b = fix_vectors(a, b)
    return np.subtract(a, b)


def subtract_abs(a, b):
    return np.absolute(subtract(a,b))


def multiply(a, b):
    a, b = fix_vectors(a, b)
    return np.multiply(a, b)


def protectedDiv(a, b):
    a, b = fix_vectors(a, b)
    x = []
    for i in range(len(a)):
        if b[i] == 0:
            x.append(1)
        else:
            x.append(a[i]/b[i])
    return x


def concat(a, b):
    return np.concatenate((a, b))


def min_v(a, b):
    a, b = fix_vectors(a, b)
    x = []
    for i in range(len(a)):
        x.append(min(a[i], b[i]))
    return x


def max_v(a, b):
    a, b = fix_vectors(a, b)
    x = []
    for i in range(len(a)):
        x.append(max(a[i], b[i]))
    return x


def if_v(a, b, c):
    if type(a) == float:
        a = [a]
    if type(b) == float:
        b = [b]
    if type(c) == float:
        c = [c]

    l = min(len(a), len(b), len(c))
    x = []
    for i in range(l):
        if a[i] >= 0:
            x.append(b[i])
        else:
            x.append(c[i])

    return x

