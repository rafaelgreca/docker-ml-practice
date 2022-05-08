import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

seed=23
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    config = tf.config.set_visible_devices([], 'GPU')

class LSTM_Model:

    def __init__(self, df, path, embedding_size, epochs, max_len, dropout_rate, batch_size, lstm_units):
        self.model = None
        self.df = df
        self.tokenizer = None
        self.max_len = max_len
        self.vocab_size = 0
        self.X_train, self.X_validation, self.y_train, self.y_validation = (None, None, None, None)
        self.score = 0
        self.path = path
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_units = lstm_units

    def build(self):
        self.__preprocess_data()
        self.model = keras.Sequential()
        self.model.add(layers.Embedding(input_dim=self.vocab_size,
                                        output_dim=self.embedding_size,
                                        input_length=self.max_len,
                                        name='embedding_layer'))
        self.model.add(layers.LSTM(units=self.lstm_units,
                                   name='lstm_layer'))
        self.model.add(layers.Dropout(rate=self.dropout_rate,
                                      name='dropout_layer'))
        self.model.add(layers.Dense(units=1,
                                    activation='sigmoid',
                                    name='dense_layer'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self):
        callback = EarlyStopping(patience=2)
        self.model.fit(self.X_train,
                       self.y_train,
                       batch_size=self.batch_size, 
                       epochs=self.epochs, 
                       validation_data=(self.X_validation, self.y_validation),
                       callbacks=[callback])

    def evaluate(self):
        predicted_validation = (self.model.predict(self.X_validation) > 0.5).astype("int32")
        score = f1_score(self.y_validation, predicted_validation, average='weighted')
        score = round(score, 5)
        self.score = score
        return score

    def prediction(self, test_df):
        test_df['tokenized'] = self.tokenizer.texts_to_sequences(test_df['tweet'].values)

        X = pad_sequences(sequences=test_df['tokenized'],
                          padding='pre',
                          maxlen=self.max_len)

        predicted_validation = (self.model.predict(X) > 0.5).astype("int32")
        test_df['label'] = predicted_validation
        test_df = test_df.drop(columns=['tokenized'])
        test_df.to_csv('{}/data/test_df_prediction.csv'.format(self.path), index=False, sep=',')

    def __preprocess_data(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.df['tweet'])
        self.df['tokenized'] = self.tokenizer.texts_to_sequences(self.df['tweet'].values)
        self.vocab_size = len(self.tokenizer.word_index) + 1

        X = pad_sequences(sequences=self.df['tokenized'],
                          padding='pre',
                          maxlen=self.max_len)
        y = self.df['label']

        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(X,
                                                                                            y,
                                                                                            test_size=0.25,
                                                                                            random_state=seed,
                                                                                            shuffle=False)