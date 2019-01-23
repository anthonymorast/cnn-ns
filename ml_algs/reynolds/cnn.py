from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard

"""
    Returns the model with time added in the ANN
"""
def get_model1():
    # https://stackoverflow.com/questions/47818968/adding-an-additional-value-to-a-convolutional-neural-network-input
    model = Sequential()
    # input_shape=(28, 28, 1) = width, height, channels[1=greyscale, 3=rgb]
    # kernel_size = size of kernel, 3=3x3
    model.add(Conv2D(64, kernel_size=3, activation=relu, input_shape=input_size))
    model.add(Conv2D(32, kernel_size=3, activation=relu))
    model.add(flatten())
    model.add(Dense(10, activation='softmax'))

    # add time parameter
    time = Sequential()
    time.add(Dense(1,input_shape=(1,), activation='softmax'))

    merged = Concatenate([model, time])
    # add output layer
    # compile merged model

    return merged

"""
    Returns a CNN with time added as an extra channel or the model with padded
    images. The input_size should be the only thing that differs between these
    two models.
"""
def get_model2(input_size=(160,350,3)):
    model = Sequential()

    # was good....
    model.add(Conv2D(128, kernel_size=5, activation='relu', input_shape=input_size))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(4,4)))

    # model.add(Conv2D(128, kernel_size=5, activation='relu', input_shape=input_size))
    # model.add(Conv2D(128, kernel_size=3, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(5,5), dim_ordering='th'))
    # model.add(Conv2D(128, kernel_size=3, activation='relu'))
    # model.add(Conv2D(128, kernel_size=3, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(4,4), dim_ordering='th'))
    # model.add(Conv2D(64, kernel_size=3, activation='relu'))
    # model.add(Conv2D(64, kernel_size=3, activation='relu'))
    # model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    for _ in range(9):
        model.add(Dense(10, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    # print(model.summary)

    return model
