import h5py
import pandas as pd
import numpy as np
from ann import *

if __name__ == '__main__':
    filename = 'numerical_data.h5'

    trainx = pd.read_hdf(filename, key='train_x').as_matrix()
    trainy = pd.read_hdf(filename, key='train_y').as_matrix()
    testx = pd.read_hdf(filename, key='test_x').as_matrix()
    testy = pd.read_hdf(filename, key='test_y').as_matrix()

    # train = pd.read_csv('numerical_train.csv')
    # test = pd.read_csv('numerical_test.csv')
    #
    # trainy = np.atleast_2d(train['re'].as_matrix()).T
    # trainx = train.loc[:, train.columns != 're'].as_matrix()
    # testy = np.atleast_2d(test['re'].as_matrix()).T
    # testx = test.loc[:, test.columns != 're'].as_matrix()

    # buld model
    # num_layers = 15
    # sizes = [150000, 100000, 75000, 50000, 25000, 20000,
    #          15000, 10000, 7000, 4000, 2000, 1000, 500, 100, 20]
    # sizes = [100 for _ in range(num_layers)]
    # model = ANN(input_size=trainx.shape[1], num_hidden_layers=num_layers, hidden_layer_sizes=sizes,
    #               output_size=1, epochs=250, batch_size=32, fit_verbose=1, optimizer='adam')
    # model.build_model()
    # checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='mean_absolute_error', save_best_only=True, mode='min')
    # tensorboard = TensorBoard(log_dir='./tf_logs', histogram_freq=10, write_graph=True, write_grads=True, write_images=True)
    # model.train(trainx, trainy, validation_data=(testx, testy), callbacks=[checkpoint, tensorboard])

    # load model from file
    model = ANN()
    model.load_model('numerical_all/15x100NN_all/nn15x100_ep195.h5')

    y_hat = model.predict(testx)
    predictions = open("predictions.dat", 'w')
    testy = testy.tolist()
    for i in range(len(testy)):
        predictions.write(str(y_hat[i][0])+','+str(testy[i][0])+'\n')
    predictions.close()
