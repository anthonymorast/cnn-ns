import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # example: https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
    filename = 'numerical_data.h5'

    # need to rewrite the h5 files, these will be useful for that
    trainx = pd.read_hdf(filename, key='train_x')
    trainy = pd.read_hdf(filename, key='train_y')
    testx = pd.read_hdf(filename, key='test_x')
    testy = pd.read_hdf(filename, key='test_y')

    # save these for later
    testrows = testx.shape[0]
    trainrows = trainx.shape[0]

    ## only use training data for PCA (if we're ethical)
    ## https://www.quora.com/Why-should-PCA-only-be-fit-on-the-training-set-and-not-the-test-set
    pca_x = trainx.as_matrix()
    # https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    pca_x = StandardScaler().fit_transform(pca_x)

    ## https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    ## Less Memory: https://stackoverflow.com/questions/43357507/pca-memory-error-in-sklearn-alternative-dim-reduction
    components = 61 ### 61 components explained 99.5% of the variance (as seen in output file)
    model = PCA(n_components=components)
    model.fit_transform(pca_x)

    ## https://stackoverflow.com/questions/32857029/python-scikit-learn-pca-explained-variance-ratio-cutoff
    print(model.explained_variance_ratio_)
    print(model.explained_variance_ratio_.cumsum())

    # create new datasets
    colnames = ['component'+str(i) for i in range(0, components)]
    new_trainx = model.transform(pca_x)
    new_testx = StandardScaler().fit_transform(testx.as_matrix())
    new_testx = model.transform(new_testx)

    trainx = pd.DataFrame(new_trainx, columns=colnames)
    testx = pd.DataFrame(new_testx, columns=colnames)


    testx.to_csv('numerical_test.csv', index=False)
    trainx.to_csv('numerical_train.csv', index=False)
    testy.to_csv('testy.csv', index=False)
    trainy.to_csv('trainy.csv', index=False)
