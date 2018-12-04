import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    basedir = os.path.join('..', 'tools', 'numerical', 'data')

    test_x = None
    test_y = None
    train_x = None
    train_y = None
    data = None
    for file in os.listdir(basedir):
        print("Processing: ", file)
        path = os.path.join(basedir, file, 'data.csv')
        df = pd.read_csv(path, low_memory=False)
        if data is None:
            data = df
        else:
            data = data.append(df)
        df = df.drop(df.index, inplace=True)

    print(data.columns, data.shape)
    data = data.sample(frac=1).reset_index(drop=True)
    # other ways to do this:
    #  https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
    train = data.sample(frac=0.8)
    test = data.drop(train.index)

    train_x = train.iloc[:, train.columns != 're']
    train_y = train.iloc[:, train.columns == 're']

    test_x = test.iloc[:, test.columns != 're']
    test_y = test.iloc[:, test.columns == 're']

    print(test_x.shape, test_y.shape, train_x.shape, train_y.shape)
    print(len(train), len(train_x), len(train_y), len(test), len(test_x), len(test_y))

    hd5_file = 'data.h5'
    train_x.to_hdf(hd5_file, key='train_x')
    train_y.to_hdf(hd5_file, key='train_y')
    test_x.to_hdf(hd5_file, key='test_x')
    test_y.to_hdf(hd5_file, key='test_y')
