"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset1, make_dataset2
from plot import plot_boundary
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    train_set = make_dataset2(1200, 565354)
    test_set = make_dataset2(300, 156)
    seeds = [5, 36, 47, 9898]
    depth_test = [1, 2, 4, 8]
    scores = {}

    # create tree and figure of unconstrained depth

    estimator = DecisionTreeClassifier().fit(train_set[0], train_set[1])
    plot_boundary("inf_train_tree", estimator, train_set[0], train_set[1])
    plot_boundary("inf_test_tree", estimator, test_set[0], test_set[1])
    prediction = estimator.predict(test_set[0])
    scores[0] = []
    scores[0].append(accuracy_score(test_set[1], prediction))

    # part 2, test model against 5 test set.
    for seed in seeds:
        test_set = make_dataset2(300, seed)
        prediction = estimator.predict(test_set[0])
        scores[0].append(accuracy_score(test_set[1], prediction))

    # create tree and figure for each depth
    for depth in depth_test:
        print("create tree" + str(depth))
        estimator = DecisionTreeClassifier(
            max_depth=depth). fit(train_set[0], train_set[1])
        plot_boundary("test_tree_" + str(depth),
                      estimator, test_set[0], test_set[1])
        plot_boundary("train_tree_" + str(depth),
                      estimator, train_set[0], train_set[1])
        prediction = estimator.predict(test_set[0])
        scores[depth] = []
        scores[depth].append(accuracy_score(test_set[1], prediction))

        # part 2 test over 5 generation

        for seed in seeds:
            test_set = make_dataset2(300, seed)
            prediction = estimator.predict(test_set[0])
            scores[depth].append(accuracy_score(test_set[1], prediction))

        # print result

    for depth in scores:
        print(" for depth: " + str(depth) + " mean: " +
              str(np.mean(scores[depth])) + " std: " + str(np.std(scores[depth])))
