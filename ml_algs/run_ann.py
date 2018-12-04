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

    num_layers = 100
    sizes = [145000, 130000, 110000, 85000, 60000, 50000, 50000,
             35000, 30000, 25000, 25000, 25000, 20000, 15000, 10000,
             50000, 2500, 1500, 750, 200]
    sizes = [50 for _ in range(num_layers)]
    # for i in range(40):
    #     sizes.append(75000)
    # for i in range(29):
    #     sizes.append(50000)
    # for i in range(20):
    #     sizes.append(20000)
    # for i in range(5):
    #     sizes.append(10000)
    # for i in range(4):
    #     sizes.append(1000)
    # sizes.append(50)
    model = ANN(trainx.shape[1], num_hidden_layers=num_layers, hidden_layer_sizes=sizes,
                  output_size=1, epochs=500, batch_size=64, fit_verbose=1)
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='mean_absolute_error', save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir='./tf_logs', histogram_freq=10, write_graph=True, write_grads=True, write_images=True)
    model.train(trainx, trainy, validation_data=(testx, testy), callbacks=[checkpoint, tensorboard])

    y_hat = model.predict(testx)
    predictions = open("predictions.dat", 'w')
    testy = test.tolist()
    for i in range(len(testy)):
        predictions.write(str(y_hat[i][0])+','+str(testy[i])+'\n')
    predictions.close()
