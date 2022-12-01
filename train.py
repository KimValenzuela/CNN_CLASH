import json
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from preprocess import Preprocess
from model import inceptionv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

plt.switch_backend('agg')
K.set_image_data_format('channels_last')

def get_params(dataset):
    with open(f"./{dataset}/config.json", "r") as f:
        params = json.load(f)
    return params

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis = -1))

def generator(nb_batches_train, train_data, dataset):
    while True:
        for i in range(nb_batches_train):
            samples = np.random.choice(len(train_data), 128)
            train_data_epoch = train_data.iloc[samples]
            preprocess = Preprocess(train_data_epoch, f'{dataset}/stamps/')
            X, y = preprocess.get_data()

            datagen = ImageDataGenerator(
                rotation_range=360,
                width_shift_range=0.05,
                height_shift_range=0.05,
                horizontal_flip=True,
                vertical_flip=True
            )
            data = datagen.flow(X, y, batch_size=X.shape[0], shuffle=False).next()
            yield data[0], data[1]


def plot_learning_curve(loss, val_loss, score):
    plt.clf()
    plt.plot(loss, color='k')
    plt.plot(val_loss, color='b')
    plt.axhline(score, linestyle='--', color='r')
    plt.title('model RMSE')
    plt.ylabel('RMSE')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation','Test'], loc='upper right')
    plt.plot(len(loss)-1, score, '*', color='r', markersize=10)
    plt.savefig("lossvsepoch.png")


if __name__ == '__main__':
    dataset = str(sys.argv[1])

    data = pd.read_csv(f'{dataset}/{dataset}_labels.csv', index_col=None)
    data = data.sample(frac=1).reset_index(drop=True)

    params = get_params(dataset)

    val_size = params['val_size']
    test_size = params['test_size']

    val_data = data[:val_size].reset_index(drop=True)
    test_data = data[val_size:val_size+test_size].reset_index(drop=True)
    train_data = data[val_size+test_size:].reset_index(drop=True)

    model = inceptionv2(params['image_size'])
    model.compile(loss=root_mean_squared_error, optimizer=Adam(learning_rate=params['learning_rate']))

    early_stopping = EarlyStopping(monitor='val_loss', patience=params['early_stopping'], mode='auto')

    nb_batches_train = int(train_data.shape[0]/params['batch_size_train'])

    prep_val = Preprocess(val_data, f'{dataset}/stamps/')
    prep_test = Preprocess(test_data, f'{dataset}/stamps/')

    history = model.fit(
        generator(nb_batches_train, train_data, dataset), 
        steps_per_epoch=nb_batches_train, epochs=params['num_epochs'], 
        validation_data=prep_val.get_data(), callbacks=[early_stopping]
    )

    X_test, y_test = prep_test.get_data()
    score = model.evaluate(X_test, y_test)
    print('Score: ', score)
    
    plot_learning_curve(history.history['loss'], history.history['val_loss'], score)
            
