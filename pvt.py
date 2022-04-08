import numpy as np
import os
import cv2
import argparse
import tflite_runtime.interpreter as tflite


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                        help="path of configuration yaml file"),
parser.add_argument("--model_path", type=str, default="ONNX/",
                        help="path of onnx extra data folder"),
parser.add_argument("--shape_predictor", type=str, default="preprocess/shape_predictor_68_face_landmarks.dat",
                        help="path of z_id tensor")
parser.add_argument("--z_id_path", type=str, default="preprocess/z_id.npy",
                        help="path of z_id tensor"),
parser.add_argument("--target_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000002.png",
                        help="path of preprocessed target face image"),

args = parser.parse_args()

#load data
img = cv2.imread(args.target_image)
img = np.expand_dims(img, axis=0)
print (img.shape)
img = np.transpose(img, (0, 3, 1, 2))
print (img.shape)

z_id = np.load(args.z_id_path)
z_id = np.uint8(z_id)


interpreter = tflite.Interpreter(args.model_path+ "MultiLevelEncoder_gen_Lite_optimized.tflite", num_threads=24)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], img)

interpreter_ADD = tflite.Interpreter(args.model_path+ "ADD_gen_Lite_optimized.tflite", num_threads=24)
interpreter_ADD.allocate_tensors()

input_details_ADD = interpreter_ADD.get_input_details()
output_details_ADD = interpreter_ADD.get_output_details()

interpreter.invoke()

#print(interpreter.get_tensor(output_details[0]['index']))

z2 = interpreter.get_tensor(output_details[0]['index'])
z1 = interpreter.get_tensor(output_details[1]['index'])
z8 = interpreter.get_tensor(output_details[2]['index'])
z3 = interpreter.get_tensor(output_details[3]['index'])
z6 = interpreter.get_tensor(output_details[4]['index'])
z4 = interpreter.get_tensor(output_details[5]['index'])
z5 = interpreter.get_tensor(output_details[6]['index'])
z7 = interpreter.get_tensor(output_details[7]['index'])

interpreter_ADD.set_tensor(input_details_ADD[0]['index'], z5)
interpreter_ADD.set_tensor(input_details_ADD[1]['index'], z_id)
interpreter_ADD.set_tensor(input_details_ADD[2]['index'], z6)
interpreter_ADD.set_tensor(input_details_ADD[3]['index'], z2)
interpreter_ADD.set_tensor(input_details_ADD[4]['index'], z1)
interpreter_ADD.set_tensor(input_details_ADD[5]['index'], z3)
interpreter_ADD.set_tensor(input_details_ADD[6]['index'], z7)
interpreter_ADD.set_tensor(input_details_ADD[7]['index'], z8)
interpreter_ADD.set_tensor(input_details_ADD[8]['index'], z4)

interpreter_ADD.invoke()

output_image = interpreter_ADD.get_tensor(output_details_ADD[0]['index'])
print(output_image[0].shape)
#output_image[0] = np.transpose(output_image[0], (1, 2, 0))
out = np.zeros((256,256,3))
out[:,:,0] = output_image[0][0]
out[:,:,1] = output_image[0][1]
out[:,:,2] = output_image[0][2]

cv2.imshow("out", out)
cv2.waitKey(0)

"""
output Multilevel
0 [1 2048 4 4]
1 [1 1024 2 2]
2 [1  64 256 256]
3 [1 1024 8 8]
4 [1 128  64  64]
5 [1 512  16  16]
6 [1 256  32  32]
7 [1  64 128 128]
"""

"""
input add
[  1 256  32  32]
[  1 256]
[  1 128  64  64]
[   1 2048    4    4]
[   1 1024    2    2]
[   1 1024    8    8]
[  1  64 128 128]
[  1  64 256 256]
[  1 512  16  16]
"""