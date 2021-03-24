from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
from sklearn.ensemble import  RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score


def fetchDataset(dataset='TrainOnMe.csv'):
    df = pd.read_csv(dataset)
    df = df.dropna(axis=0, how='any')

    df = df.replace(['Flase', 'Bayesian Interference'], ['False', 'Bayesian Inference'])
    df = df.replace(['True', 'False', 'GMMs and Accordions', 'Bayesian Inference'], [1, -1, 2, -2])
    # df_test = df.sample(frac=0.3)
    # df = df[~df.index.isin(df_test.index)]
    x_type = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']
    y = df['y'].to_numpy().T
    x = pd.DataFrame(df, columns=x_type)
    x = x.values

    # yte = df_test['y'].to_numpy().T
    # xte = pd.DataFrame(df_test, columns=x_type)
    # xte = xte.values
    # return x, y, xte, yte
    return x, y



# def pre_processing(X, y):
#     pca = PCA(n_components=12)
#     pca.fit(X)
#     X = pca.transform(X)
#     return X, y


def fetch_evaluate(dataset='EvaluateOnMe.csv'):
    df = pd.read_csv(dataset)
    x_type = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']
    df1 = df.replace([True, False, 'GMMs and Accordions', 'Bayesian Inference'], [1, -1, 200, -200])
    x = pd.DataFrame(df1, columns=x_type)
    x = x.values
    return x


def output_evaluate(x, y):
    output_array = np.concatenate((y, x), axis=1)
    df = pd.DataFrame(output_array, columns= ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'])
    return df
if __name__ == '__main__':
    # X, y, xte, yte = fetchDataset()
    X, y= fetchDataset()
    # X, y = pre_processing(X, y)

    clf_random_forest = RandomForestClassifier(n_estimators=200)
    # scores_random_forest = cross_val_score(clf_random_forest, X, y, cv=5)
    # print(scores_random_forest.mean())
    clf_random_forest.fit(X=X, y=y)

    # score = clf_random_forest.score(X=xte, y=yte)
    # print(score)

    # predict = clf_random_forest.predict(X=xte)
    # count = 0.0
    # for i in range(len(yte)):
    #     if yte[i] == predict[i]:
    #         count += 1
    # print(count/len(yte))

    evaluate_x = fetch_evaluate()
    predict = clf_random_forest.predict(evaluate_x)
    prediction_df = pd.DataFrame(predict)
    prediction_df.to_csv('output.txt', index=False, header=False)

