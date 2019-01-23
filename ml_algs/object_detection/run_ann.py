import h5py
import pandas as pd
import numpy as np
from ann import *

from sklearn.metrics import mean_absolute_error, mean_squared_error

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

    # load model from file
    model = ANN()
    model.load_model('./all/20x100/best.h5')

    ## buld model
    #num_layers = 20
    #sizes = [50 for _ in range(num_layers)]
    #model = ANN(input_size=trainx.shape[1], num_hidden_layers=num_layers, hidden_layer_sizes=sizes,
    #             output_size=4, epochs=5000, batch_size=256, fit_verbose=1, optimizer='adam')
    #model.build_model()
    
    #checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='loss', save_best_only=True, mode='min')
    #tensorboard = TensorBoard(log_dir='./tf_logs', histogram_freq=10, write_graph=True, write_grads=True, write_images=True)
    #cb = [checkpoint]#,tensorboard]
    #model.train(trainx, trainy, validation_data=(testx, testy), callbacks=cb)

    y_hat = model.predict(testx)
    print("Test MAE:", mean_absolute_error(y_hat, testy))
    print("Test MSE:", mean_squared_error(y_hat, testy))
    y_hat = model.predict(trainx)
    print("Train MAE:", mean_absolute_error(y_hat, trainy))
    print("Train MSE:", mean_squared_error(y_hat, trainy))
    
    #predictions = open("predictions.dat", 'w')
#    x1 = 0
#    x2 = 0
#    y1 = 0
#    y2 = 0
#   count = 0
#    for i in range(testy.shape[0]):
        #predictions.write(str(y_hat[i])+','+str(testy[i])+'\n')
#        x1 += abs(testy[i][0] - y_hat[i][0])
#        x2 += abs(testy[i][1] - y_hat[i][1])
#        y1 += abs(testy[i][2] - y_hat[i][2])
#        y2 += abs(testy[i][3] - y_hat[i][3])
#        count += 1
#    print("x1:", (x1/count), "x2:", (x2/count), "y1:", (y1/count), "y2:", (y2/count))
    #predictions.close()
