import onnx
from onnx import helper
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                    help="path of configuration yaml file")
parser.add_argument("--checkpoint_path", type=str, default="chkpt/30.ckpt",
                    help="path of aei-net pre-trained file")
parser.add_argument("--model_path", type=str, default="ONNX/single/",
                    help="path of output onnx"),
parser.add_argument("--target_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000011.png",
                    help="path of preprocessed target face image"),
parser.add_argument("--source_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000056.png",
                    help="path of preprocessed source face image"),
parser.add_argument("--output_image", type=str, default="output.png",
                    help="path of output image")
parser.add_argument("--gpu_num", type=int, default=0,
                    help="number of gpu")
args = parser.parse_args()

model = onnx.load(args.model_path)
intermediate_layer_value_info = helper.ValueInfoProto()
intermediate_layer_value_info.name = intermediate_tensor_name
model.graph.output.append(intermediate_layer_value_info)
onnx.save(model, model_path)