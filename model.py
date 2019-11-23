# Importing the libraries
from numpy.random import seed
seed(1)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import pickle

survey = pd.read_csv('Resources/winequality-red.csv')


X = survey.drop("quality", axis=1)
y = survey["quality"]


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1, stratify=y, train_size=0.75, test_size=0.25)

X_scaler = MinMaxScaler().fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
encoded_y_train = label_encoder.transform(y_train)
encoded_y_test = label_encoder.transform(y_test)

y_train_categorical = to_categorical(encoded_y_train)
y_test_categorical = to_categorical(encoded_y_test)

model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=11))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
# model.add(Dense(units=100, activation='relu'))
# model.add(Dense(units=100, activation='relu'))
# model.add(Dense(units=100, activation='relu'))
# model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=6, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    X_train_scaled,
    y_train_categorical,
    epochs=100,
    shuffle=True,
    verbose=2
)

model_loss, model_accuracy = model.evaluate(
    X_test_scaled, y_test_categorical, verbose=2)


# Saving model to disk
model.save('redwinequality_model_trained.h5')

# Loading model to compare the results
from keras.models import load_model
survey_model = load_model('redwinequality_model_trained.h5')