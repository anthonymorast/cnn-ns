"""
    More abstraction for Keras (which is obviously needed). This class can create and
    train a model or load a model or model weights from hdf5 filesselfself.

    Examples:
    1. Create a model:
        num_layers = 15
        sizes = [100 for _ in range(num_layers)]
        model = ANN(input_size=trainx.shape[1], num_hidden_layers=num_layers, hidden_layer_sizes=sizes,
                      output_size=1, epochs=250, batch_size=32, fit_verbose=1, optimizer='adam')
        model.build_model()

    2. Load a model
        model = ANN()
        model.load_model('numerical_all/15x100NN_all/nn15x100_ep195.h5')
"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard


class ANN(object):
    def __init__(self, input_size=None, num_hidden_layers=0, hidden_layer_sizes=[], output_size=0,
                 epochs=50, batch_size=1, fit_verbose=2, variables=None, optimizer='adam'):
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = fit_verbose
        self.optimizer = optimizer

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_layer_sizes[0], input_shape=(self.input_size, ),
                             activation='relu'))
        for i in range(1, self.num_hidden_layers - 1):
            self.model.add(Dense(self.hidden_layer_sizes[i], activation='relu'))
        self.model.add(Dense(self.hidden_layer_sizes[len(self.hidden_layer_sizes) - 1], activation='relu'))
        self.model.add(Dense(self.output_size, activation='relu'))
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['mae'])

    def load_model(self, filename):
        """
            Loads a Keras model from an HDF5 file.
        """
        self.model = load_model(filename)

    def predict(self, data):
        """
            Runs the data in the data parameter through the network and
            returns a list of predicted values.

             data - a matrix of data (explanatory variables) to be sent through the LSTM
        """
        return self.model.predict(data)


    def get_weights(self):
        """
            Returns the weights for each layer in the network (list of arrays).
        """
        return self.model.get_weights()


    def set_weights(self, weights):
        """
            Sets the weights of the network.
        """
        self.model.set_weights(weights)


    def train(self, train_x, train_y, optimzer='adam', callbacks=[], validation_data=()):
        """
            Trains the model using the Adam optimization algortihm (more to be implemented
            later). Creates a 'history' attr of the LSTM.

            train_x - a matrix of explanatory variables for training
            train_y - a matrix of dependent variables to train on
            optimizer - optimization algorithm (Adam is the only one implemented)
         """
        self.history = self.model.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size,
                                      verbose=self.verbose, callbacks=callbacks, validation_data=validation_data)

    def get_model(self):
        """
            Returns the model. Allows adding callbacks, etc.
        """
        return self.model
