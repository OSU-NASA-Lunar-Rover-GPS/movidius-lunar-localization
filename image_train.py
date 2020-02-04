from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

tf.keras.backend.clear_session()

# (raw_train, raw_validation, raw_test), metadata = tfds.load(
#     'cats_vs_dogs',
#     split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
#     with_info=True,
#     as_supervised=True,
# )

get_label_name = metadata.features['label'].int2str

for image, label in raw_train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

inputs = tf.keras.Input(shape=(2, 224, 224, 3))

# load resnet50
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
feature_batch = base_model(image_batch)

# freeze layers
base_model.trainable = False

features_list = [layer.output for layer in base_model.layers]

feat_extraction_model = tf.keras.Model(inputs=base_model.input, outputs=features_list)

reprojection_features = feat_extraction_model(inputs[0])
aerial_features = feat_extraction_model(inputs[1])

combined_features = tf.stack(reprojection_features, aerial_features, axis=1)
combined_features = tf.reshape(combined_features, [-1, 2 * 2048])

dense_layer = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
dense_batch = dense_layer()

dropout_layer = tf.keras.layers.Dropout(rate=0.3)
dropout_batch = dropout_layer(dense_batch)

prediction_layer = tf.keras.layers.Dense(units=2)
prediction_batch = prediction_layer(dropout_batch)

model = tf.keras.Sequential([
  base_model,
  dropout_layer,
  prediction_layer
])