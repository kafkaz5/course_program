import pandas as pd
import numpy as np
import lightgbm as lgb

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
    return x, y


if __name__ == '__main__':
    x, y = fetchDataset()
    train_data = lgb.Dataset(x, y)
    validation_data = lgb.Dataset(reference=train_data)
    w = np.random.rand(len(x),)
    train_data.set_weight(w)
    param = {'num_leaves': 31, 'objective': 'binary'}
    param['metric'] = ['auc', 'binary_logloss']
    num_round = 10
    bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
