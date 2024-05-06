#!/usr/bin/env python
# coding: utf-8
import os
import hls4ml
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from performance_test import evaluate_performance, render_results
from model_setup import build_model, build_config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

np.random.seed(16)
tf.keras.utils.set_random_seed(16)

config = build_config({
    "name": "StandardTensorflow",
    "load_saved_model": False,
    "model_shape": Sequential([
        Input(name="Input", shape=(28, 28)),
        Flatten(name="Flatten"),
        QDense(32, name="QDense1", kernel_quantizer=quantized_bits(2, 0), bias_quantizer=quantized_bits(2, 0)),
        QActivation(quantized_relu(2)),
        QDense(10, name="QDense2", kernel_quantizer=quantized_bits(2, 0), bias_quantizer=quantized_bits(2, 0)),
        Activation("softmax", name="Softmax")
    ]),
    "compilation": {
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"]
    },
    "fit": {
        "epochs": 1,
        "batch_size": 32
    }
})

model = build_model(config)

performances = evaluate_performance(model, config["test_data"], config["test_labels"])
render_results(performances)

# granularity can be "model", "type", or "name"

config = hls4ml.utils.config_from_keras_model(model, granularity="model", backend="VivadoAccelerator")
config["Model"]["ReuseFactor"] = 1000000
config["Model"]["Strategy"] = "Resource"
config["Model"]["Precision"] = "fixed<2,0>"

print(json.dumps(config, indent=2))

OUTPUT_DIR = "build"

hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=OUTPUT_DIR, backend="VivadoAccelerator")

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

hls_model.compile()
# TODO: check accuracy here with hls_model.predict()
# TODO: What does compile actually do?

#print("Let's build!")
# os.chdir('/mnt/c/Users/peter/Code/Studium/test/build')
# os.system('vivado_hls -f build_prj.tcl "reset=False csim=False synth=True cosim=False validation=False export=True vsynth=False fifo_opt=False"')
# os.system('vivado -mode batch -source design.tcl')
# hls_model.build(csim=False, export=True, bitfile=True)
# TODO: What does build actually do? What does the export argument do? What about csim and bitfile?

# hls4ml.report.read_vivado_report(OUTPUT_DIR)
