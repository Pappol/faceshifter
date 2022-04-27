import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="ONNX/model.onnx",
                    help="path of onnx file to convert")
parser.add_argument("--output_path", type=str, default="ONNX/inferred.onnx",
                    help="path of onnx output")

args = parser.parse_args()

onnx.checker.check_model(args.model_path,)
onnx.shape_inference.infer_shapes_path(args.model_path, args.output_path)