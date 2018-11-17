import glob
import numpy as np
import os
from scipy import misc
from cnn import *
import pandas as pd

def get_numpy_data(filename='probably_wont_exist.dat'):
    if os.path.isfile(filename):
        print("Loading data from", filename+"...")
        return np.load(filename)
    return None

def load_4D_data(train, test):
    train_filename = '4_channel_train'
    test_filename = '4_channel_test'

    train_data = get_numpy_data(os.path.join('.', train_filename+'.npy'))
    if train_data is None:
        print('Train data not found, creating training data...')
        train_paths = train.iloc[:,0].tolist()
        train_times = train.iloc[:,1].tolist()
        train_imgs = [misc.imread(path) for path in train_paths]
        train_imgs = np.asarray(train_imgs)
        pad = ((0,0),)*3 + ((0,1),)
        train_imgs = np.pad(train_imgs, pad, 'constant', constant_values = 0)
        dims = train_imgs.shape
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    train_imgs[i][j][k][3] = train_times[i]
        train_data = train_imgs
        np.save(train_filename, train_data)

    test_data = get_numpy_data(os.path.join('.', test_filename+'.npy'))
    if test_data is None:
        print('Test data not found, creating testing data...')
        test_paths = test.iloc[:,0].tolist()
        test_times = test.iloc[:,1].tolist()
        test_imgs = [misc.imread(path) for path in test_paths]
        test_imgs = np.asarray(test_imgs)
        pad = ((0,0), )*3 + ((0,1),)
        test_imgs = np.pad(test_imgs, pad, 'constant', constant_values=0)
        dims = test_imgs.shape
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    test_imgs[i][j][k][3] = test_times[i]
        test_data = test_imgs
        np.save(test_filename, test_data)

    return train_data, test_data

if __name__ == '__main__':
    train_path = 'train'
    test_path = 'test'
    train_file = 'train.csv'
    test_file = 'test.csv'

    train = pd.read_csv(train_file)
    # train = train.sample(frac=1).reset_index(drop=True)
    test = pd.read_csv(test_file)
    # test = test.sample(frac=1).reset_index(drop=True)

    train_x, test_x = load_4D_data(train, test)
    train_y, test_y = train.iloc[:,2], test.iloc[:,2]

    image_size = np.asarray([train_x.shape[1], train_x.shape[2], train_x.shape[3]])
    m = get_model2(image_size)

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='mean_absolute_error', save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir='./tf_logs', histogram_freq=10, write_graph=True, write_grads=True, write_images=True)
    m.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=250, callbacks=[checkpoint, tensorboard])

    y_hat = m.predict(test_x)
    predictions = open('predictions.dat', 'w')
    predictions.write('y_hat, y\n')
    test_y = test_y.tolist()
    for i in range(len(test_y)):
        predictions.write(str(y_hat[i][0])+','+str(test_y[i])+'\n')
