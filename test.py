import argparse
from PIL import Image
from omegaconf import OmegaConf
import tflite_runtime.interpreter as tflite
from imutils import face_utils
import dlib
import time
import cv2
import os
import PIL
import random
import numpy as np
import scipy.ndimage
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                        help="path of configuration yaml file"),
parser.add_argument("--model_path", type=str, default="ONNX/",
                        help="path of onnx extra data folder"),
parser.add_argument("--shape_predictor", type=str, default="preprocess/shape_predictor_68_face_landmarks.dat",
                        help="path of z_id tensor")
parser.add_argument("--z_id_path", type=str, default="preprocess/z_id.npy",
                        help="path of z_id tensor")

args = parser.parse_args()

interpreter_MultiLevelEncoder = tflite.  Interpreter(args.model_path+ "MultiLevelEncoder_gen_Lite_optimized.tflite")
multi_signature = interpreter_MultiLevelEncoder.get_signature_runner()
print (multi_signature)