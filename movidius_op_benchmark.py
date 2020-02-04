# This benchmark script is designed to evaluate the time required to load a network to the movidius device and
# the time requred to process a single inference for a specified model. One can use this script to decide whether
# the number of model loading operations should be minimized.

import os
import time
import numpy as np
from openvino.inference_engine import IENetwork, IECore, IEPlugin

MODEL_XML = "./ovino_model/model.xml"
MODEL_BIN = os.path.splitext(MODEL_XML)[0] + ".bin"

load_count = 3
inference_count = 1000

ie = IECore()
net = IENetwork(model=MODEL_XML, weights=MODEL_BIN)
plugin = IEPlugin(device="MYRIAD")
input_blob = next(iter(net.inputs))
exec_net = []

# get start time
start_time = time.time_ns()

# process 100 loads of
for i in range(load_count):
    exec_net.append(plugin.load(network=net))

# get finish time
end_time = time.time_ns()

# calculate run time
run_time = (end_time-start_time)/(load_count*1000000000)

print('Seconds per model load (averaged over ' + str(load_count) + ' load operations): ' + str(run_time)  + '\n')

# generate random image
img = np.random.random((1,3,224,224)) * 255

# get start time
start_time = time.time_ns()

# process <inference_count> inferences
for i in range(inference_count):
    res1 = exec_net[0].infer(inputs={input_blob: img})

# get finish time
end_time = time.time_ns()

# calculate run time
run_time = (end_time-start_time)/(inference_count*1000000000)

#print result
print('Seconds per inference (averaged over ' + str(inference_count) + ' inferences): ' + str(run_time)  +'\n')
