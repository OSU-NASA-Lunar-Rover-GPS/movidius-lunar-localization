#!/bin/bash

source /opt/intel/openvino/bin/setupvars.sh;

/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
  --input_model ./saved_tf_model/saved_model.pb \
  --batch 1 \
  --data_type FP16 \
  --output_dir ./ovino_model/;

