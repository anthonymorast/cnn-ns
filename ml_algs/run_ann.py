import h5py
import pandas as pd
import numpy as np
from ann import *

if __name__ == '__main__':
    filename = 'numerical_data.h5'

    # trainx = pd.read_hdf(filename, key='train_x').as_matrix()
    # trainy = pd.read_hdf(filename, key='train_y').as_matrix()
    # testx = pd.read_hdf(filename, key='test_x').as_matrix()
    # testy = pd.read_hdf(filename, key='test_y').as_matrix()

    train = pd.read_csv('numerical_train.csv')
    test = pd.read_csv('numerical_test.csv')

    trainy = np.atleast_2d(train['re'].as_matrix()).T
    trainx = train.loc[:, train.columns != 're'].as_matrix()
    testy = np.atleast_2d(test['re'].as_matrix()).T
    testx = test.loc[:, test.columns != 're'].as_matrix()

    num_layers = 25
    sizes = [30 for _ in range(num_layers)]
    model = ANN(trainx.shape[1], num_hidden_layers=num_layers, hidden_layer_sizes=sizes,
                  output_size=1, epochs=1500, batch_size=16, fit_verbose=1, optimizer='sgd')
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='mean_absolute_error', save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir='./tf_logs', histogram_freq=10, write_graph=True, write_grads=True, write_images=True)
    model.train(trainx, trainy, validation_data=(testx, testy), callbacks=[checkpoint, tensorboard])

    y_hat = model.predict(testx)
    predictions = open("predictions.dat", 'w')
    testy = testy.tolist()
    for i in range(len(testy)):
        predictions.write(str(y_hat[i][0])+','+str(testy[i])+'\n')
    predictions.close()
