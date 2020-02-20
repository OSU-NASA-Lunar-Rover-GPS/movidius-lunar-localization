### to do:
### find better solution to learning rate decay
### train_dataset_input_fn and consequently the slow dataset.shuffle() function, seem to be called after each time an evaluation is made.  Therefore...

### done:
#     preshuffle tfrecords input
#     still shuffle in input functions but only evaluate after several epochs
#     disable evaluate every 600 seconds (so that evaluation is only done once per epoch - this is because switching from train to evaluate results in the iterator being reset)
#     decay learning rate on a schedule

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# set up constants
C_INIT_LEARNING_RATE = 5e-4

EPOCHS = 100

# batches
TRAIN_BATCH_SIZE = 10
VAL_BATCH_SIZE = 10
PREDICT_BATCH_SIZE = 10*1

N_GPUS=1                         # setting to 0 uses all available GPUs
N_PREPROCESSING_THREADS=8*N_GPUS     # number of preprocessing threads
INPUT_WIDTH = 224
INPUT_HEIGHT = 224

TRAIN_TFRECORD = './reproject_truematch_allbatch_5050match_224scale_06iou.tfr'
VAL_TFRECORD = './reproject_truematch_allbatch_5050match_224scale_06iou.tfr'


def feature_model_fn(features, labels, mode):

  feature_model = tf.keras.applications.ResNet50V2(input_shape=(224, 224, 3), include_top=False, pooling='max',
                                                   classes=2048)

  tile1_images = tf.cast(features["tile1_img"], tf.float32)
  tile2_images = tf.cast(features["tile2_img"], tf.float32)

  tile1_feature_maps = feature_model(tile1_images)    # Features with shape [batch_size, num_features].
  tile2_feature_maps = feature_model(tile2_images)    # Features with shape [batch_size, num_features].

  combined_feature_map = tf.stack([tile1_feature_maps, tile2_feature_maps], axis=1) # (batch_size, 2, 1792) combined feature map
  combined_feature_map_flat = tf.reshape(combined_feature_map, [-1, 2 * 2048])

  return combined_feature_map_flat, labels


# Model function for CNN
def matching_model_fn(features, labels, mode):

  global learning_rate
  global INPUT_WIDTH, INPUT_HEIGHT

  global_step = tf.train.get_global_step()

  print('learning rate: %f' % learning_rate)

  # labels
  matches = labels['match']

  dense = tf.compat.v1.layers.dense(inputs=features, units=256, activation=tf.nn.relu)
  dropout = tf.compat.v1.layers.dropout(inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

  # logits layer
  logits = tf.compat.v1.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=matches, logits=logits)
  accuracy = tf.metrics.accuracy(labels=matches, predictions=predictions["classes"], name='acc')

  train_hook_list = []
  train_tensors_log = {'accuracy': accuracy[1],
                       'loss': cross_entropy,
                       'global_step': global_step}
  train_hook_list.append(tf.train.LoggingTensorHook(
    tensors=train_tensors_log, every_n_iter=100))
  
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(
        loss=cross_entropy,
        global_step=tf.compat.v1.train.get_global_step())
    tf.compat.v1.summary.scalar('match_loss', cross_entropy, family='train')
    return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, train_op=train_op, training_hooks=train_hook_list)

  # Add evaluation metrics (for EVAL mode)
  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {"accuracy": tf.compat.v1.metrics.accuracy(labels=matches, predictions=predictions["classes"])}
    tf.compat.v1.summary.scalar('match_loss', cross_entropy, family='eval')
    return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, eval_metric_ops=eval_metric_ops)


def _input_parser(example):
  
  features = {
    'height': tf.io.FixedLenFeature((), tf.int64),
    'width': tf.io.FixedLenFeature((), tf.int64),
    'scaled_height': tf.io.FixedLenFeature((), tf.int64),
    'scaled_width': tf.io.FixedLenFeature((), tf.int64),
    'offset_y': tf.io.FixedLenFeature((), tf.float32),
    'offset_x': tf.io.FixedLenFeature((), tf.float32),
    'match': tf.io.FixedLenFeature((), tf.int64),
    'iou': tf.io.FixedLenFeature((), tf.float32),
    'tile1_raw': tf.io.FixedLenFeature((), tf.string),
    'tile2_raw': tf.io.FixedLenFeature((), tf.string)
  }

  parsed_example = tf.io.parse_single_example(serialized=example, features=features)

  width=parsed_example['scaled_width']
  height=parsed_example['scaled_height']

  tile1_img = tf.io.decode_raw(parsed_example['tile1_raw'], tf.float64)
  tile2_img = tf.io.decode_raw(parsed_example['tile2_raw'], tf.float64)

  tile1_img = tf.image.convert_image_dtype(tile1_img, tf.float16)
  tile2_img = tf.image.convert_image_dtype(tile2_img, tf.float16)

  # reshape input images
  tile1_img = tf.reshape(tile1_img, (INPUT_HEIGHT, INPUT_WIDTH))
  tile2_img = tf.reshape(tile2_img, (INPUT_HEIGHT, INPUT_WIDTH))

  # scale intensity between 0 and 1
  tile1_img = tile1_img/255.
  tile2_img = tile2_img/255.

  # make 3-channel, grayscale image
  tile1_img = tf.stack((tile1_img, tile1_img, tile1_img), axis=2) 
  tile2_img = tf.stack((tile2_img, tile2_img, tile2_img), axis=2) 
  
  match = parsed_example['match']
  offset_x = parsed_example['offset_x']
  offset_y = parsed_example['offset_y']

  return {"tile1_img": tile1_img, "tile2_img": tile2_img, "width": width, "height": height}, {'match': match, 'offset_x': offset_x, 'offset_y': offset_y}


def train_dataset_input_fn():

  training_dataset = tf.data.TFRecordDataset(TRAIN_TFRECORD)
  training_dataset = training_dataset.map(_input_parser, num_parallel_calls=N_PREPROCESSING_THREADS)
  training_dataset = training_dataset.batch(TRAIN_BATCH_SIZE)
  training_dataset = training_dataset.repeat()
  training_dataset = training_dataset.prefetch(buffer_size=None)
  return training_dataset
 
def val_dataset_input_fn():

  SHUFFLE_BUFFER = 10000
  val_dataset = tf.data.TFRecordDataset(VAL_TFRECORD)
  val_dataset = val_dataset.map(_input_parser)
  val_dataset = val_dataset.shuffle(buffer_size=SHUFFLE_BUFFER)
  val_dataset = val_dataset.batch(VAL_BATCH_SIZE)
  val_dataset = val_dataset.repeat()

  return val_dataset

learning_rate=C_INIT_LEARNING_RATE

def main(argv):

  global learning_rate
   # Create the Estimator
  if N_GPUS:
    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=N_GPUS)
  else:
    distribution = tf.contrib.distribute.MirroredStrategy()

  config = tf.estimator.RunConfig(keep_checkpoint_max=100, keep_checkpoint_every_n_hours=0.5)
  deep_matcher = tf.estimator.Estimator(model_fn=matching_model_fn, config=config, model_dir="tf_model_v1/")

  ## training method 0 (unsure if learning persists between inner loops!)
  ## learning rate decay
  for epoch in range(EPOCHS):     # change learning rate 10 times
    for j in range(10):   # number of times to call train before changing training rate
      deep_matcher.train(input_fn=train_dataset_input_fn, steps=10000)
      deep_matcher.evaluate(input_fn=val_dataset_input_fn, steps=10)
    learning_rate = learning_rate*0.95

if __name__ == "__main__":
  tf.compat.v1.app.run()
