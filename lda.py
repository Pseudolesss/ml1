"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from plot import plot_boundary
from data import make_dataset1, make_dataset2

from sklearn.base import BaseEstimator, ClassifierMixin


class LinearDiscriminantAnalysis(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        """Fit a linear discriminant analysis model using the training set
        (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
                The training input samples.

        y : array-like, shape = [n_samples]
                The target values.

        Returns
        -------
        self : object
                Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        # ====================
        # TODO your code here.
        # ====================
        self.nb_classes = len(set(y))
        nb_classes = self.nb_classes
        nb_samples = len(y)
        prop = np.zeros(nb_classes)
        # Mean attrubute values for both class
        mean = np.zeros((nb_classes, len(X[0])))
        prop = dict()
        mean = dict()

        i = 0
        for sample in y:
            if prop.get(sample, False):  # Knowed class
                prop[sample] += 1
                mean[sample] += X[i]
            else:						# New class discovered
                prop[sample] = 1
                mean[sample] = X[i]
            i += 1

        for k in prop.keys():
        	mean[k] /= prop[k]
        	prop[k] /= nb_samples

        cov = np.cov(X, None, False)

        # Class density formula for 2 classes
        self.f = lambda x, k: ((2 * math.pi)**(nb_classes / 2) *
                               math.sqrt(np.linalg.det(cov)))**-1 * \
            np.exp(-0.5 * np.dot(np.dot((x - mean[k]),
                                        np.linalg.inv(cov)), (x - mean[k])))

        self.prop = prop

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
                The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
                The predicted classes, or the predict values.
        """

        nb_classes = self.nb_classes
        y = []
        densities = dict()
        Bayes = dict()
        den = 0

        for sample in X:

            for k in self.prop.keys():
                densities[k] = self.f(sample, k)
                den += densities[k] * self.prop[k]

            # Sending back the key with the highest value
            for k in self.prop.keys():
            	Bayes[k] = (densities[k] * self.prop[k] / den)

            y.append(max(Bayes, key=Bayes.get))

        return y

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
                The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
        The class probabilities of the input samples. Classes are ordered
        by lexicographic order.
        """

        nb_classes = self.nb_classes
        p = np.empty([len(X), nb_classes])
        num = dict()
        densities = dict()
        i = 0
        for sample in X:

            for k in self.prop.keys():
                densities[k] = self.f(sample, k)
                num[k] = densities[k] * self.prop[k]

            den = 0
            for k in self.prop.keys():
            	den += self.prop[k] * densities[k]

            numerator = list()
            for k in sorted(num.keys()):
            	numerator.append(num[k])

            numerator = np.asarray(numerator)
            p[i] = numerator / den
            i += 1

        return p


if __name__ == "__main__":

    # 1st dataset
    train_set = make_dataset1(1200, 565354)
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_set[0], train_set[1])
    plot_boundary('lda_trainDataset1', lda, train_set[0], train_set[1])

    # 2nd dataset
    train_set = make_dataset2(1200, 565354)
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_set[0], train_set[1])
    plot_boundary('lda_trainDataset2', lda, train_set[0], train_set[1])

    # Accuracy and std for five generations of different seeds

    accuracy1 = np.zeros(5)
    accuracy2 = np.zeros(5)

    seed = 10000  # Will change for each generation
    for i in range(5):
        (train_set1, test_set1) = (make_dataset1(
            1200, seed), make_dataset1(300, seed))
        (train_set2, test_set2) = (make_dataset2(
            1200, seed), make_dataset2(300, seed))

        (lda1, lda2) = (LinearDiscriminantAnalysis(),
                        LinearDiscriminantAnalysis())

        lda1.fit(train_set1[0], train_set1[1])
        lda2.fit(train_set2[0], train_set2[1])

        predict1 = lda1.predict(test_set1[0])
        predict2 = lda2.predict(test_set2[0])

        for j in range(300):
            if predict1[j] == test_set1[1][j]:
                accuracy1[i] += 1
            if predict2[j] == test_set2[1][j]:
                accuracy2[i] += 1

        seed += seed

    (mean1, mean2) = (np.mean(accuracy1), np.mean(accuracy2))
    (std1, std2) = (np.std(accuracy1), np.std(accuracy2))

    # Display of the results
    print('Dataset 1:')
    print(accuracy1)
    print('mean: ' + str(mean1) + '   std: ' + str(std1))
    print('mean%: ' + str(mean1 / 300) + '   std%: ' + str(std1 / 300))
    print()
    print('Dataset 2:')
    print(accuracy2)
    print('mean: ' + str(mean2) + '   std: ' + str(std2))
    print('mean%: ' + str(mean2 / 300) + '   std%: ' + str(std2 / 300))
