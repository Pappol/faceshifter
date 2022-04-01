import onnx
import argparse
from onnx_tf.backend import prepare
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="ONNX/single/",
                    help="path of onnx file to convert")
parser.add_argument("--ex_data", type=str, default="ONNX/",
                    help="path of onnx extra data folder")

args = parser.parse_args()

ADD_gen_onnx = onnx.load(args.model_path + "ADD_gen.onnx")
MultilevelEncoder_onnx = onnx.load(args.model_path + "MultilevelEncoder.onnx")

#converting from onnx to TF

tf_rep_ADD = prepare(ADD_gen_onnx)
tf_rep_Multi = prepare(MultilevelEncoder_onnx)

tf_rep_ADD.export_graph(args.ex_data + "ADD_gen")
tf_rep_Multi.export_graph(args.ex_data + "MultilevelEncoder")

#converting from TF to TFLite

converter_ADD = tf.lite.TFLiteConverter.from_saved_model(args.ex_data + "ADD_gen")
tflite_model_ADD = converter_ADD.convert()

converter_Multi = tf.lite.TFLiteConverter.from_saved_model(args.ex_data + "MultilevelEncoder")
tflite_model_Multi = converter_Multi.convert()

# Save the models

with open(args.ex_data + "ADD_gen_Lite.h5", 'wb') as f:
    f.write(tflite_model_ADD)

with open(args.ex_data + "MultilevelEncoder_Lite.h5", 'wb') as f:
    f.write(tflite_model_Multi)
