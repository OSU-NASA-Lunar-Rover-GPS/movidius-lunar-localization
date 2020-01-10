### to do:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import tensorflow as tf
import tensorflow_hub as hub

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

PREDICT_BATCH_SIZE = 10
LMBDA = 100.
INPUT_WIDTH = 224
INPUT_HEIGHT = 224


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""

    module_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1"
    tfhub_module = hub.Module(module_url)

    global INPUT_WIDTH, INPUT_HEIGHT
    tile1_images = tf.cast(features["tile1_img"], tf.float32)
    tile2_images = tf.cast(features["tile2_img"], tf.float32)

    # reshape input images
    tile1_images = tf.reshape(tile1_images, (-1, INPUT_HEIGHT, INPUT_WIDTH))
    tile2_images = tf.reshape(tile2_images, (-1, INPUT_HEIGHT, INPUT_WIDTH))

    # make 3-channel, grayscale image
    tile1_images = tf.stack((tile1_images, tile1_images, tile1_images), axis=3)
    tile2_images = tf.stack((tile2_images, tile2_images, tile2_images), axis=3)

    tile1_feature_maps = tfhub_module(tile1_images)  # Features with shape [batch_size, num_features].
    tile2_feature_maps = tfhub_module(tile2_images)  # Features with shape [batch_size, num_features].

    combined_feature_map = tf.stack([tile1_feature_maps, tile2_feature_maps],
                                    axis=1)  # (batch_size, 2, 1792) combined feature map

    # fc layer
    combined_feature_map_flat = tf.reshape(combined_feature_map, [-1, 2 * 2048])
    dense = tf.compat.v1.layers.dense(inputs=combined_feature_map_flat, units=256, activation=tf.nn.relu)
    dropout = tf.compat.v1.layers.dropout(
        inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    # logits layer
    logits = tf.compat.v1.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        # "offsets": offsets
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


deep_matcher = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="models/")


def _input_parser(example):
    features = {
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'offset_y': tf.io.FixedLenFeature((), tf.float32),
        'offset_x': tf.io.FixedLenFeature((), tf.float32),
        'match': tf.io.FixedLenFeature((), tf.int64),
        'iou': tf.io.FixedLenFeature((), tf.float32),
        'tile1_raw': tf.io.FixedLenFeature((), tf.string),
        'tile2_raw': tf.io.FixedLenFeature((), tf.string)
    }

    parsed_example = tf.io.parse_single_example(serialized=example, features=features)
    width = parsed_example['width']
    height = parsed_example['height']
    tile1_img = tf.io.decode_raw(parsed_example['tile1_raw'], tf.uint8)
    tile2_img = tf.io.decode_raw(parsed_example['tile2_raw'], tf.uint8)

    match = parsed_example['match']
    offset_x = parsed_example['offset_x']
    offset_y = parsed_example['offset_y']

    metadata = {
        'offset_y': offset_y,
        'offset_x': offset_x,
        'match': parsed_example['match'],
        'iou': parsed_example['iou'],
        'width': width,
        'height': height}
    return {"tile1_img": tile1_img, "tile2_img": tile2_img, "width": width, "height": height}, {'match': match,
                                                                                                'offset_x': offset_x,
                                                                                                'offset_y': offset_y}


def predict_dataset_input_fn():
    global PREDICT_BATCH_SIZE
    PREDICT_TFRECORD = 'reproject_truematch_allbatch_5050match_224scale_06iou.tfr'

    predict_dataset = tf.data.TFRecordDataset(PREDICT_TFRECORD)
    predict_dataset = predict_dataset.map(_input_parser)
    predict_dataset = predict_dataset.batch(PREDICT_BATCH_SIZE)
    # return predict_dataset
    predict_it = tf.compat.v1.data.make_one_shot_iterator(predict_dataset)

    features, labels = predict_it.get_next()
    return features, labels


def main(argv):
    # Create the Estimator
    deep_matcher = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="models/")

    predictions = deep_matcher.predict(
        input_fn=predict_dataset_input_fn)  # do inference on the dataset (set the dataset up in predict_dataset_input_fn
    for prediction in predictions:
        print(prediction)
    exit()


if __name__ == "__main__":
    tf.compat.v1.app.run()
