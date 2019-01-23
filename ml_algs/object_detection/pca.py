import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    hf = h5py.File('traindata.h5', 'r')
    data = hf.get('all')[()]
    hf.close()
    
    trainy = np.atleast_2d(data[:, range(281250, 281254)]) ### NOTE: IMPORTANT NOTE: the y data format is (x1, x2, y1, y2)
    trainx = np.atleast_2d(data[:, range(0, 281250)])
    
    hf = h5py.File('testdata.h5', 'r')
    data = hf.get('all')[()]
    hf.close()
    
    testy = data[:, range(281250, 281254)] ### NOTE: IMPORTANT NOTE: the y data format is (x1, x2, y1, y2)
    testx = data[:, range(0, 281250)]

    ## only use training data for PCA (if we're ethical)
    ## https://www.quora.com/Why-should-PCA-only-be-fit-on-the-training-set-and-not-the-test-set
    pca_x = trainx
    
    ### row 1330 was nan........
    #print(data.shape)
    #data = data[~np.isnan(data).any(axis=1)]
    #h5f = h5py.File('traindata.h5', 'w')
    #h5f.create_dataset('all', data=data)
    
    # https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    pca_x = StandardScaler(copy=False).fit_transform(pca_x)
    
    ## https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    ## Less Memory: https://stackoverflow.com/questions/43357507/pca-memory-error-in-sklearn-alternative-dim-reduction
    components = 400 ### 400 components explained 97.99% of the variance (as seen in output file)
    model = TruncatedSVD(n_components=components, algorithm='arpack') # PCA -> memory issues
    model.fit(pca_x)

    ## https://stackoverflow.com/questions/32857029/python-scikit-learn-pca-explained-variance-ratio-cutoff
    print(model.explained_variance_ratio_)
    print(model.explained_variance_ratio_.cumsum())

    # create new datasets
    colnames = ['component'+str(i) for i in range(0, components)]
    new_trainx = model.transform(pca_x)
    new_testx = StandardScaler().fit_transform(testx)
    new_testx = model.transform(new_testx)
    
    h5f = h5py.File('testx_rot.h5', 'w')
    h5f.create_dataset('all', data=new_testx)
    
    h5f = h5py.File('trainx_rot.h5', 'w')
    h5f.create_dataset('all', data=new_trainx)
    
    h5f = h5py.File('testy_rot.h5', 'w')
    h5f.create_dataset('all', data=testy)
    
    h5f = h5py.File('trainy_rot.h5', 'w')
    h5f.create_dataset('all', data=trainy)
