"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from plot import plot_boundary
from data import make_dataset1, make_dataset2
from sklearn.model_selection import StratifiedKFold

# (Question 2)


if __name__ == "__main__":
    n_table = [1, 5, 25, 125, 300, 625,1200]
    data = make_dataset2(1500, 1997)
    scores = {}
    mean = {}
    var = {}
    for n in n_table:

        # part 1
        estimator = KNeighborsClassifier(n_neighbors=n).fit(data[0], data[1])
        print("computing" + str(n))
        plot_boundary("knn_" + str(n), estimator, data[0], data[1])

        scores[n] = cross_val_score(
            estimator, data[0], data[1], cv=10).tolist()
        for i in range(9):
            cv = StratifiedKFold(n_splits=10, random_state=i, shuffle=True)
            scores[n].extend(cross_val_score(
                estimator, data[0], data[1], cv=cv).tolist())
        print(len(scores[n]))
        mean[n] = np.mean(scores[n])
        var[n] = np.var(scores[n])
    print("mean" + str(mean))
    print("var" + str(var))

    # part2
    # desired accuracy
    delta_mean = 0.01
    delta_var = delta_mean**2

    # create 4 points of interest equally spaced.
    points = list(range(4))
    points[0] = 1
    points[3] = 1300
    points[1] = points[0] + (points[3] - points[0]) // 3
    points[2] = points[3] - (points[3] - points[0]) // 3

    # compute cross mean/vars of cross score validation.
    while points[0] < points[1] < points[2] < points[3]:
        print(points)
        for n in points:
            estimator = KNeighborsClassifier(
                n_neighbors=n).fit(data[0], data[1])
            scores[n] = cross_val_score(
                estimator, data[0], data[1], cv=10).tolist()
            for i in range(9):
                cv = StratifiedKFold(n_splits=10, random_state=i, shuffle=True)
                scores[n].extend(cross_val_score(
                    estimator, data[0], data[1], cv=cv).tolist())
            mean[n] = np.mean(scores[n])
            var[n] = np.var(scores[n])
        if abs(mean[points[0]] - mean[points[3]]) < delta_mean and abs(var[points[0]] - var[points[3]]) < delta_var:
            print(str(points[0]) + " to " + str(points[3]))
            break
        # Find the best K among the 4 point of interest.
        best_var = var[points[0]]
        best_mean = mean[points[0]]
        best_k = 0
        for i in range(4):
            if abs(mean[points[0]] - mean[points[3]]) < delta_mean:
                # best point is less variance.
                if var[points[i]] < best_var:
                    best_var = var[points[i]]
                    best_k = i
            else:
                # best point is max mean.
                if mean[points[i]] > best_mean:
                    best_mean = mean[points[i]]
                    best_k = i
        # update the point of interest
        print("var: " + str(best_var) + " mean: " + str(best_mean))
        print(best_k)
        if best_k == 0:
            points[3] = points[1]
        elif best_k == 3:
            points[0] = points[2]
        else:
            points[0] = points[best_k - 1]
            points[3] = points[best_k + 1]
        points[1] = points[0] + (points[3] - points[0]) // 3
        points[2] = points[3] - (points[3] - points[0]) // 3
