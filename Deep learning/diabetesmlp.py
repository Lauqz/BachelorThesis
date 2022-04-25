import tensorflow as tf
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split

batch_size = 128
num_classes = 2
epochs = 2000

diabetes = pd.read_csv('datasets/diabetes2.CSV')
diabetes.describe()

X = diabetes.iloc[:,:8].values
y = diabetes.iloc[:,8].values

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
X_train = X[:576, :]
X_test = X[576:, :]
y_train = y[:576]
y_test = y[576:]

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(12, activation='relu',input_shape=(8,)))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(), metrics = ['accuracy']) 
	     

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test,y_test))

score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
