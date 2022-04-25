import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape = (224,224,3))
for layer in vgg_model.layers[:-5]:
	layer.trainable=False
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='auto', min_delta=0.01)

model = tf.keras.models.Sequential()
model.add(vgg_model)
model.add(tf.keras.layers.Flatten(input_shape=vgg_model.output_shape[1:]))
model.add(tf.keras.layers.Dropout(0.8))
model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(149, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001, decay=0.0001/200), loss='categorical_crossentropy', metrics=['accuracy'])

train= tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test= tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
training_set = train.flow_from_directory('datasets/generation/train', target_size=(224,224), class_mode = 'categorical')
val_set = test.flow_from_directory('datasets/generation/test', target_size=(224,224), class_mode = 'categorical')

history = model.fit_generator(training_set, steps_per_epoch = 64, epochs = 500, validation_data = val_set, validation_steps = 64, callbacks=[es])

N = es.stopped_epoch-1 if es.stopped_epoch != 0 else 500
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), history.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), history.history["val_acc"], label="val_acc")
plt.title("Training loss and accuracy on dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss/Acc")
plt.legend(loc="lower left")
plt.savefig("plot.png") 

test_image = tf.keras.preprocessing.image.load_img('datasets/starter/val/attempt.png', target_size=(224, 224))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(training_set.class_indices)
print(result)
