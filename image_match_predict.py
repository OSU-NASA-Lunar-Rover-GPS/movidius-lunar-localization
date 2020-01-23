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
CAMERA_DEVICE_NUMBER = 0


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


class gui(tk.Tk):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.parent = parent

        self.topframe = tk.Frame(self.parent)
        self.topframe.pack(side="top",expand=False)

        self.bottomframe = tk.Frame(self.parent)
        self.bottomframe.pack(side="top",expand=False)

        self.panelA = None
        self.panelB = None
        self.panelC = None
        self.panelD = None
        #self.panelE = None
        self.cam_img = [None, None, None, None]

    def capture_location(self):

        # initialize camera array

        for i in range(4):
            # capture image
            cam = cv2.VideoCapture(CAMERA_DEVICE_NUMBER)
            ret, frame = cam.read()
            cam.release()

            # convert to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ### post process
            # convert to PIL
            self.cam_img[i] = Image.fromarray(frame)

            # resize for interface
            self.cam_img[i] = self.cam_img[i].resize((320, 180), Image.ANTIALIAS)

            # convert to ImageTk
            self.cam_img[i] = ImageTk.PhotoImage(self.cam_img[i])

            time.sleep(1)

        # create interface

        if self.panelA is None or self.panelB is None or self.panelC is None or self.panelD is None:
            self.panelA = tk.Label(self.topframe, image=self.cam_img[0])
            self.panelA.image = self.cam_img[0]
            self.panelA.pack(side="left", padx=10, pady=10)

            self.panelB = tk.Label(self.topframe, image=self.cam_img[1])
            self.panelB.image = self.cam_img[1]
            self.panelB.pack(side="left", padx=10, pady=10)

            self.panelC = tk.Label(self.topframe, image=self.cam_img[2])
            self.panelC.image = self.cam_img[2]
            self.panelC.pack(side="left", padx=10, pady=10)

            self.panelD = tk.Label(self.topframe, image=self.cam_img[3])
            self.panelD.image = self.cam_img[3]
            self.panelD.pack(side="left", padx=10, pady=10)

            # self.panelE = tk.Label(image=reprojection)
            # self.panelE.image = reprojection
            # self.panelE.pack(side="right", padx=10, pady=10)
            button = tk.Button(self.bottomframe, text="Capture Location", command=self.capture_location)
            button.pack(fill=tk.X, padx=10, pady=10)
        else:
            # update the pannels
            self.panelA.configure(image=self.cam_img[0])
            self.panelB.configure(image=self.cam_img[1])
            self.panelC.configure(image=self.cam_img[2])
            self.panelD.configure(image=self.cam_img[3])
            # self.panelE.configure(image=reprojection)
            self.panelA.image = self.cam_img[0]
            self.panelB.image = self.cam_img[1]
            self.panelC.image = self.cam_img[2]
            self.panelD.image = self.cam_img[3]
            # self.panelE.image = reprojection


def main():

    # parse arguments
    args = build_argparser().parse_args()

    root = tk.Tk()
    root.title("Lunar Localization Program")
    main_gui = gui(root)

    main_gui.capture_location()

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
