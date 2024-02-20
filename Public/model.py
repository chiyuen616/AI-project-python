import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,4.5) # Make the default figures a bit bigger
import numpy as np
import random
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


import yfinance as yf

# get historical market data from y_finance
def AnomalyDetectorModel(stockName): 

    HSI = yf.Ticker(stockName)
    HSI = HSI.history(period="max")
    HSI["OCAverage"] = HSI[["Open", "Close"]].mean(axis=1)
    HSI = HSI["OCAverage"]
    HSI = HSI.dropna()
    HSI.index = HSI.index.strftime('%m/%d/%Y')

    HSI.plot()

# def the class of Anomaly Detection
    class AnomalyDetector(Model):
        def __init__(self):
            super(AnomalyDetector, self).__init__()
            self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")])

            self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(30, activation="sigmoid")])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    autoencoder = AnomalyDetector()

    # apply SMAPE Loss function instead of MAE and MSE
    def smape_loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        denominator = tf.keras.backend.abs(y_true) + tf.keras.backend.abs(y_pred)
        diff = tf.keras.backend.abs(y_true - y_pred)
        smape = 2.0 * tf.keras.backend.mean(diff / denominator, axis=-1) * 100
        return smape

    autoencoder.compile(optimizer='adam', loss=smape_loss)

    # normalize data using MinMaxScaler, window size 30
    def normalize_window_roll(data, window_size):
        X = []
        Y = []
        for i in range(0,len(data) - window_size,1):
            x_window = data.iloc[i:i+window_size]

            # Normalize the x_window and y_window
            scaler = MinMaxScaler()
            x_window = scaler.fit_transform(np.array(x_window).reshape(-1, 1))

            X.append(x_window)
        X =np.squeeze(np.array(X), axis=2)

        return X

    X= normalize_window_roll(HSI,30)

    # divide data into 3 parts, 70% train, 10% test, 20% test
    train_size = int(0.7*X.shape[0])
    test_size = int(0.1*X.shape[0])
    val_size = int(0.2*X.shape[0])

    X_train = X[:train_size]
    X_test = X[train_size:train_size+test_size]
    X_val = X[train_size+test_size:train_size+test_size+val_size]

    # fit data to train and set epochs and early stoppage
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=2,
                                                    mode='min')
    autoencoder.compile(optimizer='adam', loss=smape_loss)

    history = autoencoder.fit(X_train,X_train, epochs=400,
                        validation_data=(X_val, X_val),
                        batch_size=16,
                        callbacks=[early_stopping])

    # calculate and plot the train loss
    reconstructions = autoencoder.predict(X_train)
    train_loss = smape_loss(reconstructions, X_train)

    plt.hist(train_loss[None,:], bins=50)
    plt.xlabel("Train loss")
    plt.ylabel("No of examples")
    plt.show()

    # define and calculate the threshold
    threshold = np.mean(train_loss) + np.std(train_loss) 
    print("Threshold: ", threshold)

    # calculate and plot the test loss
    reconstructions = autoencoder.predict(X_test)
    test_loss = smape_loss(reconstructions, X_test)

    plt.hist(test_loss[None, :], bins=50)
    plt.xlabel("Test loss")
    plt.ylabel("No of examples")
    plt.show()

    # define anomalous data
    def predict(model, data, threshold):
        reconstructions = model(data)
        loss = smape_loss(reconstructions, data)
        return tf.math.less(loss, threshold)

    preds = predict(autoencoder, X_test, threshold)
    anomalous_test_data = X_test[np.where(preds==False)]

    # plot the graph to show the anomalous
    encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()

    for p in [1,10,20]:
        plt.plot(anomalous_test_data[p], 'b')
        plt.plot(np.arange(0,X.shape[1]),decoded_data[p], 'r')
        plt.fill_between(np.arange(X.shape[1]), decoded_data[p], anomalous_test_data[p], color='lightcoral')
        plt.legend(labels=["Input", "Reconstruction", "Error"])
        plt.show()

