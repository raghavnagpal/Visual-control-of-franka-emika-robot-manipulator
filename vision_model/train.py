from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import numpy as np

from tensorflow.keras import datasets, layers, models

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import matplotlib.pyplot as plt

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# j = tf.test.is_gpu_available(
#     cuda_only=False, min_cuda_compute_capability=None
# )
# print("is cuda ", j)


def make_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(7, activation='tanh'))
    # model.add(layers.Multiply(3.14))

    model.summary()

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=['mae', 'mse'])
    return model


model = make_model()

counter = 0
input_file = "dataset_3d_train.npz"
# input_file = "dataset_test.npz"

data = np.load(input_file)

# dataset numpy
img_array = data['img']
gt_tx = data['gt_tx']

SHUFFLE = True
if SHUFFLE:
    s = np.arange(img_array.shape[0])
    np.random.shuffle(s)

    img_array = img_array[s]
    gt_tx = gt_tx[s]

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
# tf.compat.v1.keras.backend.get_session

checkpoint_path = 'model/model.{epoch:05d}-{val_loss:.6f}.h5'

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 verbose=1)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001)

EPOCHS = 100

history = model.fit(
    img_array, gt_tx,
    batch_size=32,
    epochs=EPOCHS,
    validation_split=0.1,
    verbose=1,
    callbacks=[tfdocs.modeling.EpochDots(), cp_callback])



# Save the model
model.save('model/end_model_1.h5')
print("model saved")

example_batch = img_array[:10]
example_batch_mask =  img_array[:10]
example_batch_test = gt_tx[:10]
example_result = model.predict(example_batch_mask)
print(example_batch_test)
print(example_result)

print("end")
