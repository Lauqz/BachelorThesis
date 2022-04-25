import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=(200,200,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=400, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

training_set= train.flow_from_directory('datasets/starter/train', target_size=(200,200), class_mode='categorical')
val_set= test.flow_from_directory('datasets/starter/test', target_size=(200,200), class_mode='categorical')

history=model.fit_generator(training_set, steps_per_epoch=16, epochs=10, validation_data=val_set, validation_steps=16)

test_image = tf.keras.preprocessing.image.load_img('datasets/starter/val/attempt.png', target_size=(200, 200))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
print(training_set.class_indices)
print(result)

test_image2 = tf.keras.preprocessing.image.load_img('datasets/starter/val/attempt2.png', target_size=(200, 200))
test_image2 = tf.keras.preprocessing.image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis=0)
result2 = model.predict(test_image2)
print(training_set.class_indices)
print(result2)
