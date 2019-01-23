from sklearn import tree
import pandas as pd
import numpy as np

if __name__ == '__main__':
    train = pd.read_csv('numerical_train.csv')
    test = pd.read_csv('numerical_test.csv')

    trainy = np.atleast_2d(train['re'].as_matrix()).T
    trainx = train.loc[:, train.columns != 're'].as_matrix()
    testy = np.atleast_2d(test['re'].as_matrix()).T
    testx = test.loc[:, test.columns != 're'].as_matrix()

    t = tree.DecisionTreeRegressor(presort=True, criterion='friedman_mse')
    t = t.fit(trainx, trainy)

    y_hat = t.predict(testx)
    predictions = open("tree_predictions.dat", 'w')
    testy = testy.tolist()
    sum = 0
    for i in range(len(testy)):
        predictions.write(str(y_hat[i])+','+str(testy[i])+'\n')
        sum += abs(y_hat[i] - testy[i])
    print("MAE:", (sum/len(y_hat)))
    predictions.close()
