import onnx
import argparse
from onnx_tf.backend import prepare
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from IPython.display import display
import torch

def compare_models(args):

    ADD_gen_onnx = onnx.load(args.model_path + "ADD_gen.onnx")
    MultilevelEncoder_onnx = onnx.load(args.model_path + "MultilevelEncoder.onnx")

    #converting from onnx to TF

    tf_rep_ADD = prepare(ADD_gen_onnx)
    tf_rep_Multi = prepare(MultilevelEncoder_onnx)

    img = Image.open(args.target_image)
    img = np.transpose(img, (2, 0, 1))

    img = np.expand_dims(img, axis=0)
    img = img/255.0
    img = img.astype(np.float32)
    output = tf_rep_Multi.run(img)
    print("The type is : ", type(output))
    

    z_id = np.load(args.z_id_path).astype(np.float32)
    #input = [output[4], z_id, output[5], output[1], output[0], output[2], output[6], output[7], output[3]]
    #input = [output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7]]

    #input = [z_id, output[5], output[6], output[7], output[1], output[2], output[3], output[0], output[4]]
    input = [output[4], z_id, output[5], output[1], output[0], output[2], output[6], output[7], output[3]]
    image = tf_rep_ADD.run(input)

def main(args):

    compare_models(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--z_id_path", type=str, default="../preprocess/z_id.npy",
                            help="path of z_id tensor"),
    parser.add_argument("--target_image", type=str, default="../data/faceshifter-datasets-preprocessed/train/00000003.png",
                            help="path of preprocessed target face image"),
    parser.add_argument("--output_path", type=str, default="../",
                    help="path to the output image"),
    parser.add_argument("--model_path", type=str, default="../ONNX/single/",
                    help="path of onnx file to convert")

    args = parser.parse_args()

    main(args)