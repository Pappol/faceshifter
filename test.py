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
import time

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                        help="path of configuration yaml file"),
parser.add_argument("--model_path", type=str, default="ONNX/",
                        help="path of onnx extra data folder"),
parser.add_argument("--shape_predictor", type=str, default="preprocess/shape_predictor_68_face_landmarks.dat",
                        help="path of z_id tensor")
parser.add_argument("--z_id_path", type=str, default="preprocess/z_id.npy",
                        help="path of z_id tensor"),
parser.add_argument("--target_image", type=str, default="data/00000002.png",
                        help="path of preprocessed target face image"),

args = parser.parse_args()

start_time = time.time()


img = cv2.imread(args.target_image)
img = cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)
aa = np.int8(img)

interpreter = tflite.  Interpreter(args.model_path+ "MultiLevelEncoder_gen_Lite_optimized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], aa)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

interpreter_ADD = tflite.Interpreter(args.model_path+ "ADD_gen_Lite_optimized.tflite")
interpreter_ADD.allocate_tensors()

input_details_ADD = interpreter_ADD.get_input_details()
output_details_ADD = interpreter_ADD.get_output_details()

input_shape = input_details_ADD[0]['shape']
input_data_ADD = np.array(np.random.random_sample(input_shape), dtype=np.int8)
interpreter_ADD.set_tensor(input_details_ADD[0]['index'], input_data_ADD)

interpreter_ADD.invoke()

print("--- %s seconds ---" % (time.time() - start_time))
