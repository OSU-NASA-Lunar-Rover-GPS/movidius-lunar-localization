#!/bin/bash

source /opt/intel/openvino/bin/setupvars.sh;

/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
  --input_meta_graph ./tf_model/model.ckpt-370000.meta \
  --batch 1 \
  --data_type FP16 \
  --output_dir ./ovino_model/;

# output_size = ((input_size + 2*padding - filter_size)/stride) + 1
