import pandas as pd
import numpy as np
import tensorflow as tf
import os
import datetime
import tensorboard
import tensorflow as tf
import tensorflow_hub as hub
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
import keras
import keras.backend as kback
from keras.utils import to_categorical
import re
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json
import pickle
import sys

# for running from command line, if -warm then uses pre trained weights
load_weights = False
if len(sys.argv) > 1:
    args = str(sys.argv[1:])
    if '-warm' in args:
        load_weights = True

# print out devices available (to see if there is a GPU)
print(device_lib.list_local_devices())

# set the system path
system_path = os.getcwd()

# set callback  dir
log_dir = system_path + '\\.logs'
model_file_path = 'models/toxic_comment_DNN.h5'
weights_file_path = "models/weights.best.hdf5"
tokenizer_file_path = 'models/tokenizer.pkl'

# load raw data
data = pd.read_csv('data/train.csv')
text_col = 'comment_text'
target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# set up logging for use with tensorbord
tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False)

# split train and test data
X_train_text, X_test_text, y_train, y_test = train_test_split(data[text_col], np.array(data[target_cols]),
                                                              test_size=0.3, random_state=0)
# set up the tokenizer
num_words = 2000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X_train_text.tolist())
X_train = tokenizer.texts_to_matrix(X_train_text)
X_test = tokenizer.texts_to_matrix(X_test_text)

# get parameters used to setup the model architecture
num_classes = y_test.shape[1]
num_dims = X_train.shape[1]

# set up the model check point to save only the best weights
checkpoint = ModelCheckpoint(weights_file_path, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='max')

# Create the model
model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(num_dims, )))
model.add(Dropout(0.3))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid'))
if load_weights:
    print('loading pre trained wieghts')
    model.load_weights(weights_file_path, by_name=False)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train,
          y_train,
          batch_size=800,
          shuffle=True,
          epochs=1,
          validation_data=(X_test, y_test),
          callbacks=[tensorboard, checkpoint,  EarlyStopping(min_delta=0.001, patience=2)])

# Evaluate the model
scores = model.evaluate(X_test, y_test)

print('Accuracy: ', scores[1])

# save model arch in json, note best weights have already been saved as part of the checkpoint
model_json = model.to_json()
pickle.dump(model_json, open(model_file_path, 'wb'))

# saving the tokenizer
with open(tokenizer_file_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading the tokenizer
with open(tokenizer_file_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# load the model json
with open(model_file_path, 'rb') as f:
    model_json = pickle.load(f)

# create the model from json and load the wieghts
model_loaded = model_from_json(model_json)
model_loaded.load_weights(weights_file_path, by_name=False)

# predict the test set
preds = model_loaded.predict(X_test)

# print out AUC by class for all the models.
for i, j in enumerate(target_cols):
    r = roc_auc_score(np.array(y_test)[:, i], preds[:, i])
    print(j, ' auc: ', r)
print('completed')