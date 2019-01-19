import h5py
import pandas as pd
import numpy as np
from ann import *

if __name__ == '__main__':
    
    ## Load Data
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
    
    #hf = h5py.File('trainx_rot.h5', 'r')
    #trainx = hf.get('all')[()]
    #hf = h5py.File('trainy_rot.h5', 'r')
    #trainy = hf.get('all')[()]
    
    #hf = h5py.File('testx_rot.h5', 'r')
    #testx = hf.get('all')[()]
    #hf = h5py.File('testy_rot.h5', 'r')
    #testy = hf.get('all')[()]
    
    print("Size check:", trainx.shape, trainy.shape, testx.shape, testy.shape)

    ## buld model
    num_layers = 100
    sizes = [100 for _ in range(num_layers)]
    model = ANN(input_size=trainx.shape[1], num_hidden_layers=num_layers, hidden_layer_sizes=sizes,
                  output_size=4, epochs=500, batch_size=128, fit_verbose=1, optimizer='adam')
    model.build_model()
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='mean_absolute_error', save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir='./tf_logs', histogram_freq=10, write_graph=True, write_grads=True, write_images=True)
    cb = [checkpoint]#,tensorboard]
    model.train(trainx, trainy, validation_data=(testx, testy), callbacks=cb)

    # load model from file
    #model = ANN()
    #model.load_model('numerical_all/15x100NN_all/nn15x100_ep195.h5')

    y_hat = model.predict(testx)
    print(y_hat.shape)
    predictions = open("predictions.dat", 'w')
    for i in range(testy.shape[0]):
        predictions.write(str(y_hat[i])+','+str(testy[i])+'\n')
    predictions.close()
