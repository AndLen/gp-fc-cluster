import csv
import multiprocessing
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import base
from deap import creator
from deap import gp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score

import ea_simple_elitism
import multi_tree as mt
import vector_tree as vt
from ParallelToolbox import ParallelToolbox
from read_data import read_data
from selection import *

POP_SIZE = 1024
NGEN = 100
CXPB = 0.8
MUTPB = 0.2
ELITISM = 10
MAX_HEIGHT = 8

REP = mt  # individual representation {mt (multi-tree) or vt (vector-tree)}
N_TREES = 7
DATA_DIR = '.'  # "/home/schofifinn/PycharmProjects/SSResearch/data"

METRIC = 'intra'
MAXIMISE = METRIC != 'intra'


def connectedness(cluster):
    print(cluster)


def evaluate(individual, toolbox, data, k, metric, maximise, distance_vector=None, labels_true=None):
    """
    Evaluates an individuals fitness. The fitness is the clustering performance on the data
    using the metric.

    :param individual: the individual to be evaluated
    :param toolbox: the evolutionary toolbox
    :param data: the data to be used to evaluate the individual
    :param k: the number of clusters for k-means
    :param metric: the metric to be used to evaluate clustering performance
    :param distance_vector: a pre-computed distance vector, required for silhouette-pre metric
    :param labels_true: the ground truth cluster labels, required for ari metric
    :return: the fitness of the individual
    """

    X = REP.process_data(individual, toolbox, data)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=k, random_state=SEED).fit(X)
    labels = kmeans.labels_

    # individuals that find single cluster are unfit
    nlabels = len(set(labels))
    if nlabels == 1:
        print(maximise)
        return [np.inf] if maximise else [-1]

    # uses precomputed distances of original data to avoid trending towards single dimension to
    # minimise silhouette.
    if metric == 'silhouette_pre':
        if distance_vector is None:
            raise Exception("Must provide distance vector for silhouette-pre metric.")
        return [silhouette_score(distance_vector, labels, metric='precomputed')]
    elif metric == 'silhouette':
        return [silhouette_score(X, labels, metric='euclidean')]
    elif metric == 'ari':
        if labels_true is None:
            raise Exception("Must provide ground truth labels for ARI")
        return [adjusted_rand_score(labels_true, labels)]
    elif metric == 'intra':
        sum = 0.
        for l in np.unique(labels):
            pts = data[labels == l]
            avg = pts.mean(axis=0)
            diff = np.linalg.norm(pts - avg)
            sum += diff
        return [sum]
        # raise NotImplementedError("intra not implemented")
    elif metric == 'connectedness':
        sum = 0.
        for l in np.unique(labels):
            idxs = labels == l
            dists = distance_vector[idxs][:, idxs]
            # not a single point in a cluster by itself
            if dists.shape[0] > 1:
                dists = np.divide(1., dists, out=np.zeros_like(dists), where=dists != 0)
                dists = np.minimum(dists, 10)
                np.fill_diagonal(dists, 0)
                sum += dists[dists > 0].sum() / dists.shape[0]  # .mean()
        return [sum / nlabels]
    else:
        raise Exception("invalid metric: {}".format(metric))


def write_ind_to_file(ind, run_num, results):
    """
    Writes the attributes of an individual to file.

    :param run_num: the number of the current run
    :param ind: the individual
    :param results: a dictionary of results, titles to values
    """

    line_list = []

    # add constructed features to lines
    if REP is mt:
        for cf in [str(tree) for tree in ind]:
            line_list.append(cf + "\n")
    elif REP is vt:
        for cf in vt.parse_tree(ind):
            line_list.append(cf + "\n")
    else:
        raise Exception("Invalid representation")

    line_list.append("\n")

    fl = open("%d_ind.txt" % run_num, 'w')
    fl.writelines(line_list)
    fl.close()

    csv_columns = results.keys()
    csv_file = "%d_results.txt" % run_num

    with open(csv_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerow(results)


def init_toolbox(toolbox, pset):
    """
    Initialises the toolbox with operators.

    :param toolbox: the toolbox to initialise
    :param pset: primitive set for evolution
    """
    REP.init_toolbox(toolbox, pset)
    toolbox.register("select", selElitistAndTournament, tournsize=7, elitism=ELITISM)


def init_stats():
    """
    Initialises a MultiStatistics object to capture data.

    :return: the MultiStatistics object
    """
    fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats = tools.MultiStatistics(fitness=fitness_stats)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    return stats


def final_evaluation(best, data, labels, num_classes, toolbox, print_output=True):
    """
    Performs a final performance evaluation on an individual.

    :param best: the individual to evaluate
    :param data: the dataset associated with the individual
    :param labels: the ground-truth labels of the dataset
    :param num_classes: the number of classes of the dataset
    :param toolbox: the evolutionary toolbox
    :param print_output: whether or not to print output
    :return: a dictionary of results, titles to values
    """
    kmeans = KMeans(n_clusters=num_classes, random_state=SEED).fit(data)
    labels_pred = kmeans.labels_
    baseline_ari = adjusted_rand_score(labels, labels_pred)
    baseline_silhouette = silhouette_score(data, labels_pred, metric="euclidean")

    best_ari = evaluate(best, toolbox, data, num_classes, "ari", MAXIMISE, labels_true=labels)[0]
    best_silhouette = evaluate(best, toolbox, data, num_classes, "silhouette", MAXIMISE)[0]

    if print_output:
        print("Best ARI: %f \nBaseline ARI: %f\n" % (best_ari, baseline_ari))
        print("Best silhouette: %f \nBaseline silhouette: %f" % (best_silhouette, baseline_silhouette))

    return {"best-ari": best_ari,
            "base-ari": baseline_ari, "best-sil": best_silhouette, "best-sil-pre": best.fitness.values[0],
            "base-sil": baseline_silhouette}


def plot_stats(logbook):
    gen = logbook.select("gen")
    fit_max = logbook.chapters["fitness"].select("max")
    nodes_avg = logbook.chapters["total_nodes"].select("avg")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_max, "b-", label="Maximum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, nodes_avg, "r-", label="Average Nodes")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()


rd = {}


def eval_wrapper(*args, **kwargs):
    return evaluate(*args, **kwargs, toolbox=rd['toolbox'], data=rd['data'], k=rd['num_classes'],
                    metric=rd['metric'], maximise=rd['maximise'], distance_vector=rd['distance_vector'])


# copies data over from parent process
def init_data(rundata):
    global rd
    rd = rundata


def main(datafile, run_num):
    random.seed(SEED)
    all_data = read_data("%s/%s.data" % (DATA_DIR, datafile))
    rd['data'] = all_data["data"]
    rd['labels'] = all_data["labels"]

    rd['num_classes'] = len(set(rd['labels']))
    print("%d classes found." % rd['num_classes'])
    rd['distance_vector'] = pairwise_distances(rd['data'])

    rd['num_instances'] = rd['data'].shape[0]
    rd['num_features'] = rd['data'].shape[1]
    rd['metric'] = METRIC
    rd['maximise'] = MAXIMISE
    pset = gp.PrimitiveSet("MAIN", rd['num_features'], prefix="f")
    pset.context["array"] = np.array
    REP.init_primitives(pset)
    weights = (1.0,) if MAXIMISE else (-1.,)

    creator.create("FitnessMax", base.Fitness, weights=weights)

    # set up toolbox
    toolbox = ParallelToolbox()  # base.Toolbox()
    init_toolbox(toolbox, pset)
    toolbox.register("evaluate", eval_wrapper)
    rd['toolbox'] = toolbox

    pop = toolbox.population(n=POP_SIZE)

    hof = tools.HallOfFame(1)

    stats = init_stats()
    with multiprocessing.Pool(initializer=init_data, initargs=[rd]) as pool:
        toolbox.map = pool.map
        pop, logbook = ea_simple_elitism.eaSimple(pop, toolbox, CXPB, MUTPB, ELITISM, NGEN, stats,
                                                  halloffame=hof, verbose=True)

    for chapter in logbook.chapters:
        logbook_df = pd.DataFrame(logbook.chapters[chapter])
        logbook_df.to_csv("%s_%d.csv" % (chapter, run_num), index=False)

    best = hof[0]
    res = final_evaluation(best, rd['data'], rd['labels'], rd['num_classes'], toolbox)
    write_ind_to_file(best, run_num, res)

    return pop, stats, hof


"""
[seed] [data file]
"""
if __name__ == "__main__":
    SEED = int(sys.argv[1])
    main(sys.argv[2], SEED)
