from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib.colors import ColorConverter
import random as rnd
from sklearn.datasets.samples_generator import make_blobs
from sklearn import decomposition, tree, svm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def fetchDataset(dataset='TrainOnMe.csv'):
    df = pd.read_csv(dataset)
    df = df.dropna(axis=0, how='any')
    # df = df[df['x6'].isin(['GMMs and Accordions', 'Bayesian Inference'])]
    # df = df[df['x12'].isin(['True', 'False', True, False])]
    df = df.replace([True, False, 'GMMs and Accordions', 'Bayesian Inference'], [1, -1, 2, -2])
    df = df.replace(['Shoogee', 'Atsuto', 'Bob', 'Jorg'], [0, 1, 2, 3])
    x_type = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']
    # x = df.drop(["y"], axis=1)
    y = df['y'].to_numpy().T
    x = pd.DataFrame(df, columns=x_type)
    # y = pd.DataFrame(df, columns=['y'])
    x = x.values
    # y = y.values.T
    # print(x, y)
    pcadim = 12
    return x, y, pcadim


# x, y, pcadim = fetchDataset()
#
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(x, y)
# tree.plot_tree(clf)


def plotBoundary(classifier, dataset='TrainOnMe.csv', split=0.9):
    X, y, pcadim = fetchDataset(dataset)
    xTr, yTr, xTe, yTe, trIdx, teIdx = trteSplitEven(X, y, split, 1)
    classes = np.unique(y)

    pca = decomposition.PCA(n_components=2)
    pca.fit(xTr)

    xTr = pca.transform(xTr)
    xTe = pca.transform(xTe)

    pX = np.vstack((xTr, xTe))
    py = np.hstack((yTr, yTe))

    # Train
    trained_classifier = classifier.trainClassifier(xTr, yTr)

    xRange = np.arange(np.min(pX[:, 0]), np.max(pX[:, 0]), np.abs(np.max(pX[:, 0]) - np.min(pX[:, 0])) / 100.0)
    yRange = np.arange(np.min(pX[:, 1]), np.max(pX[:, 1]), np.abs(np.max(pX[:, 1]) - np.min(pX[:, 1])) / 100.0)

    grid = np.zeros((yRange.size, xRange.size))

    for (xi, xx) in enumerate(xRange):
        for (yi, yy) in enumerate(yRange):
            # Predict
            grid[yi, xi] = trained_classifier.classify(np.array([[xx, yy]]))

    ys = [i + xx + (i * xx) ** 2 for i in range(len(classes))]
    colormap = cm.rainbow(np.linspace(0, 1, len(ys)))

    fig = plt.figure()
    # plt.hold(True)
    conv = ColorConverter()
    for (color, c) in zip(colormap, classes):
        try:
            CS = plt.contour(xRange, yRange, (grid == c).astype(float), 15, linewidths=0.25,
                             colors=conv.to_rgba_array(color))
        except ValueError:
            pass
        trClIdx = np.where(y[trIdx] == c)[0]
        teClIdx = np.where(y[teIdx] == c)[0]
        plt.scatter(xTr[trClIdx, 0], xTr[trClIdx, 1], marker='o', c=color, s=40, alpha=0.5,
                    label="Class " + str(c) + " Train")
        plt.scatter(xTe[teClIdx, 0], xTe[teClIdx, 1], marker='*', c=color, s=50, alpha=0.8,
                    label="Class " + str(c) + " Test")
    plt.legend(bbox_to_anchor=(1., 1), loc=2, borderaxespad=0.)
    fig.subplots_adjust(right=0.7)
    plt.show()


def trteSplitEven(X, y, pcSplit, seed=None):
    labels = np.unique(y)
    xTr = np.zeros((0, X.shape[1]))
    xTe = np.zeros((0, X.shape[1]))
    yTe = np.zeros((0,), dtype=int)
    yTr = np.zeros((0,), dtype=int)
    trIdx = np.zeros((0,), dtype=int)
    teIdx = np.zeros((0,), dtype=int)
    np.random.seed(seed)
    for label in labels:
        classIdx = np.where(y == label)[0]
        NPerClass = len(classIdx)
        Ntr = int(np.rint(NPerClass * pcSplit))
        idx = np.random.permutation(NPerClass)
        trClIdx = classIdx[idx[:Ntr]]
        teClIdx = classIdx[idx[Ntr:]]
        trIdx = np.hstack((trIdx, trClIdx))
        teIdx = np.hstack((teIdx, teClIdx))
        # Split data
        xTr = np.vstack((xTr, X[trClIdx, :]))
        test = y[trClIdx]
        yTr = np.hstack((yTr, y[trClIdx]))
        xTe = np.vstack((xTe, X[teClIdx, :]))
        yTe = np.hstack((yTe, y[teClIdx]))
    return xTr, yTr, xTe, yTe, trIdx, teIdx


def _testClassifier(classifier, dataset='TrainOnMe.csv', dim=12, split=0.8, ntrials=100):
    X, y, pcadim = fetchDataset(dataset)

    means = np.zeros(ntrials, );

    for trial in range(ntrials):

        xTr, yTr, xTe, yTe, trIdx, teIdx = trteSplitEven(X, y, split, trial)

        # Do PCA replace default value if user provides it
        if dim > 0:
            pcadim = dim

        if pcadim > 0:
            pca = decomposition.PCA(n_components=pcadim)
            pca.fit(xTr)
            xTr = pca.transform(xTr)
            xTe = pca.transform(xTe)

        # Train
        trained_classifier = classifier.trainClassifier(xTr, yTr)
        # Predict
        yPr = trained_classifier.classify(xTe)

        # Compute classification error
        if trial % 10 == 0:
            print("Trial:", trial, "Accuracy", "%.3g" % (100 * np.mean((yPr == yTe).astype(float))))

        means[trial] = 100 * np.mean((yPr == yTe).astype(float))

    print("Final mean classification accuracy ", "%.3g" % (np.mean(means)), "with standard deviation",
          "%.3g" % (np.std(means)))


def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts, Ndims = np.shape(X)
    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list
    # The weights for the first iteration
    wCur = np.ones((Npts, 1)) / float(Npts)
    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))
        # do classification for each point
        vote = classifiers[-1].classify(X)
        diff = np.zeros((Npts, 1))
        for i in range(Npts):
            if vote[i] == labels[i]:
                diff[i] = 1
        error = np.sum(wCur * (1 - diff))
        alpha = (np.log(1 - error) - np.log(error)) / 2
        alphas.append(alpha)
        wOld = wCur
        for i in range(Npts):
            wCur[i] = wOld[i] * np.exp(alpha * (-1) ** (vote[i] == labels[i]))
        wCur = wCur / np.sum(wCur)
    return classifiers, alphas


def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)
    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts, Nclasses))
        for i in range(Ncomps):
            hypothesis = classifiers[i].classify(X)
            for j in range(Npts):
                votes[[j], hypothesis[j]] += alphas[i]
        # one way to compute yPred after accumulating the votes
        return np.argmax(votes, axis=1)


class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


class DecisionTreeClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = DecisionTreeClassifier()
        rtn.classifier = tree.DecisionTreeClassifier(max_depth=Xtr.shape[1] / 2 + 1)
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


class svmClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = svmClassifier()
        rtn.classifier = svm.NuSVC(nu=0.4, kernel='poly', degree=12)
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


class random_forest(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = random_forest()
        rtn.classifier = RandomForestClassifier()
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)

def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts, 1)) / Npts
    else:
        assert (W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    prior = np.zeros((Nclasses, 1))
    for jdx, c in enumerate(classes):
        idx = np.where(labels == c)[0]
        prior[jdx] = np.sum(W[idx]) / Npts
    return prior


def mlParams(X, labels, W=None):
    assert (X.shape[0] == labels.shape[0])
    Npts, Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    if W is None:
        W = np.ones((Npts, 1)) / float(Npts)
    mu = np.zeros((Nclasses, Ndims))
    sigma = np.zeros((Nclasses, Ndims, Ndims))
    for jdx, c in enumerate(classes):
        idx = np.where(labels == c)[0]
        xlc = X[idx, :]
        wlc = W[idx, :]
        mu[jdx, :] = np.dot(wlc.T, xlc) / np.sum(wlc)
        normal_sigma = (np.dot(wlc.T, (xlc - mu[jdx, :]) ** 2) / np.sum(wlc))
        sigma[jdx, :, :] = np.diag(normal_sigma[0])
    return mu, sigma


def classifyBayes(X, prior, mu, sigma):
    Npts = X.shape[0]
    Nclasses, Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))
    for jdx in range(Nclasses):
        diff = X - mu[jdx]
        ln_sigma = - np.log(np.linalg.det(sigma[jdx])) / 2
        ln_diff = - 0.5 * np.diag(np.dot(diff / np.diag(sigma[jdx]), diff.T))
        ln_prior = np.log(prior[jdx])
        logProb[jdx, :] = ln_sigma + ln_diff + ln_prior
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb, axis=0)
    return h
# The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:
# NOTE: no need to touch this


class BayesClassifier(object):
    def __init__(self):
        self.trained = False
    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn
    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


if __name__ == '__main__':
    _testClassifier(classifier=random_forest())
    # x, y, pcadim = fetchDataset(dataset='TrainOnMe.csv')
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
    # gnb = GaussianNB()
    # y_pred = gnb.fit(X_train, y_train).predict(X_test)
    # print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))