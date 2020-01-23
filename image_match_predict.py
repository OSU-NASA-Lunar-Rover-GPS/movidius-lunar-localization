### to do:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import sys
import os
import time
from argparse import ArgumentParser, SUPPRESS
import numpy as np
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import cv2
import tensorflow as tf
import tensorflow_hub as hub
#from openvino.inference_engine import IENetwork, IECore, IEPlugin

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

PREDICT_BATCH_SIZE = 10
LMBDA = 100
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
MODEL_XML = "./ovino_model/model.ckpt-370000.xml"
MODEL_BIN = os.path.splitext(MODEL_XML)[0] + ".bin"
CAMERA_DEVICE_NUMBER = 2


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is MYRIAD", default="MYRIAD", type=str)

    return parser

# def _input_parser(example):
#
#     features = {
#         'height': tf.io.FixedLenFeature((), tf.int64),
#         'width': tf.io.FixedLenFeature((), tf.int64),
#         'offset_y': tf.io.FixedLenFeature((), tf.float32),
#         'offset_x': tf.io.FixedLenFeature((), tf.float32),
#         'match': tf.io.FixedLenFeature((), tf.int64),
#         'iou': tf.io.FixedLenFeature((), tf.float32),
#         'tile1_raw': tf.io.FixedLenFeature((), tf.string),
#         'tile2_raw': tf.io.FixedLenFeature((), tf.string)
#     }
#
#     parsed_example = tf.io.parse_single_example(example, features)
#     width = parsed_example['width']
#     height = parsed_example['height']
#     tile1_img = tf.io.decode_raw(parsed_example['tile1_raw'], tf.uint8)
#     tile2_img = tf.io.decode_raw(parsed_example['tile2_raw'], tf.uint8)
#
#     match = parsed_example['match']
#     offset_x = parsed_example['offset_x']
#     offset_y = parsed_example['offset_y']
#
#     metadata = {
#         'offset_y': offset_y,
#         'offset_x': offset_x,
#         'match': parsed_example['match'],
#         'iou': parsed_example['iou'],
#         'width': width,
#         'height': height}
#
#     return { "tile1_img": tile1_img,
#              "tile2_img": tile2_img,
#              "width": width,
#              "height": height}, \
#                 {   'match': match,
#                     'offset_x': offset_x,
#                     'offset_y': offset_y
#                 }



def main():

    global panelA, panelB, panelC, panelD
    panelA = None
    panelB = None
    panelC = None
    panelD = None

    # parse arguments
    args = build_argparser().parse_args()

    # create window
    root = tk.Tk()

    # initialize camera array
    cam_img = []


    for i in range(4):

        # capture image
        cam = cv2.VideoCapture(CAMERA_DEVICE_NUMBER)
        ret, frame = cam.read()
        cam.release()

        ### post process
        # convert to PIL
        cam_img.append(Image.fromarray(frame))

        # resize for interface
        cam_img[i] = cam_img[i].resize((320, 180), Image.ANTIALIAS)

        # convert to ImageTk
        cam_img[i] = ImageTk.PhotoImage(cam_img[i])

        time.sleep(3)

    # create interface

    if panelA is None or panelB is None or panelC is None or panelD is None:
        panelA = tk.Label(image=cam_img[0])
        panelA.image = cam_img[0]
        panelA.pack(side="left", padx=10, pady=10)

        panelB = tk.Label(image=cam_img[1])
        panelB.image = cam_img[1]
        panelB.pack(side="left", padx=10, pady=10)

        panelC = tk.Label(image=cam_img[2])
        panelC.image = cam_img[2]
        panelC.pack(side="left", padx=10, pady=10)

        panelD = tk.Label(image=cam_img[3])
        panelD.image = cam_img[3]
        panelD.pack(side="left", padx=10, pady=10)

        # panelE = tk.Label(image=reprojection)
        # panelE.image = reprojection
        # panelE.pack(side="right", padx=10, pady=10)
    else:
        # update the pannels
        panelA.configure(image=cam_img[0])
        panelB.configure(image=cam_img[1])
        panelC.configure(image=cam_img[2])
        panelD.configure(image=cam_img[3])
        # panelE.configure(image=reprojection)
        panelA.image = cam_img[0]
        panelB.image = cam_img[1]
        panelC.image = cam_img[2]
        panelD.image = cam_img[3]
        # panelE.image = reprojection

    ### display interface
    root.mainloop()

    # # create Inference Engine
    # ie = IECore()
    #
    # plugin = IEPlugin(device="MYRIAD")
    #
    # # load Intermediate Representation model
    # net = IENetwork(model=MODEL_XML, weights=MODEL_BIN)

    ### START CAMERA

    # open camera device 0
    # cap = cv2.VideoCapture(0)

    # take 4 pictures
    # ret, frame = cap.read()

    # generate reprojection

    ### END CAMERA

    ### START REPROJECTION MATCHING

    # predict_dataset = tf.data.TFRecordDataset('reproject_truematch_allbatch_5050match_224scale_06iou.tfr')
    #
    # predict_dataset = predict_dataset.map(_input_parser)
    # predict_dataset = predict_dataset.batch(PREDICT_BATCH_SIZE)

    # # create Inference Engine
    # ie = IECore()
    #
    # plugin = IEPlugin(device="MYRIAD")

    ## load Intermediate Representation model
    # net = IENetwork(model=MODEL_XML, weights=MODEL_BIN)
    #
    # net.batch_size = PREDICT_BATCH_SIZE
    #
    # input_blob = next(iter(net.inputs))
    # output_blob = next(iter(net.outputs))
    #
    # n, c, h, w = net.inputs[input_blob].shape
    # images = np.ndarray(shape=(n, c, h, w))
    #
    # for i in range(n):
    #     tile1_image = tf.cast(predict_dataset["tile1_img"], tf.float32)
    #
    #
    # exec_net = ie.load_network(network=net, device_name=args.device)
    # res = exec_net.infer(inputs={input_blob: images})
    # res = res[output_blob]

    ### END REPROJECTION MATCHING

    # Create the Estimator
    # deep_matcher = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="ovmodels/")

    # predictions = deep_matcher.predict(
    #     input_fn=predict_dataset_input_fn)
    # for prediction in predictions:
    #     print(prediction)
    # exit()


if __name__ == "__main__":
    sys.exit(main() or 0)
#    tf.compat.v1.app.run()
