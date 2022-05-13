import numpy as np
import os
import cv2
import argparse
import tflite_runtime.interpreter as tflite
from matplotlib import pyplot as plt
import time

def benchmark(args):
    #load data
    img = cv2.imread(args.target_image).astype(np.float32)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.transpose(2,0,1)/255.0
    img = img[np.newaxis, :]

    z_id = np.load(args.z_id_path).astype(np.float32)

    interpreter = tflite.Interpreter(args.model_path+ "MultiLevelEncoder_gen_Lite_optimized.tflite", num_threads=10)
    interpreter.allocate_tensors()
    interpreter_ADD = tflite.Interpreter(args.model_path+ "ADD_gen_Lite_optimized.tflite", num_threads=10)
    interpreter_ADD.allocate_tensors()

    interpreter_ADD = tflite.Interpreter(args.model_path+ "ADD_gen_Lite_optimized.tflite", num_threads=10)
    interpreter_ADD.allocate_tensors()

    input_details_ADD = interpreter_ADD.get_input_details()
    output_details_ADD = interpreter_ADD.get_output_details()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    for i in range(0, 10):
        start_time = time.time()

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        feature_map = []

        for i in range(0,7):
            feature_map.append(interpreter.get_tensor(output_details[i]['index']))

        print("multi --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        for i in range(0,5):
            interpreter_ADD.set_tensor(input_details_ADD[i]['index'], feature_map[i])

        interpreter_ADD.set_tensor(input_details_ADD[6]['index'], z_id)

        interpreter_ADD.set_tensor(input_details_ADD[7]['index'], feature_map[6])

        interpreter_ADD.invoke()
        print("ADD --- %s seconds ---" % (time.time() - start_time))

def test(args):
    #load data
    img = cv2.imread(args.target_image).astype(np.float32)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.transpose(2,0,1)/255.0
    img = img[np.newaxis, :]

    z_id = np.load(args.z_id_path).astype(np.float32)

    interpreter = tflite.Interpreter(args.model_path+ "MultiLevelEncoder_gen_Lite_optimized.tflite", num_threads=10)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter_ADD = tflite.Interpreter(args.model_path+ "ADD_gen_Lite_optimized.tflite", num_threads=10)
    interpreter_ADD.allocate_tensors()

    input_details_ADD = interpreter_ADD.get_input_details()
    output_details_ADD = interpreter_ADD.get_output_details()

    
    interpreter.invoke()
    """
        #print(interpreter.get_tensor(output_details[0]['index']))
        for i in range(0,7):
            print(output_details[i]['shape'])

        print('______')
        for i in range(0,8):
            print(input_details_ADD[i]['shape'])

    """

    feature_map = []

    for i in range(0,7):
        feature_map.append(interpreter.get_tensor(output_details[i]['index']))

    for i in range(0,5):
        interpreter_ADD.set_tensor(input_details_ADD[i]['index'], feature_map[i])

    interpreter_ADD.set_tensor(input_details_ADD[6]['index'], z_id)

    interpreter_ADD.set_tensor(input_details_ADD[7]['index'], feature_map[6])

    interpreter_ADD.invoke()

    output_image = interpreter_ADD.get_tensor(output_details_ADD[0]['index'])

    print(np.max(output_image))
    print(np.min(output_image))
    print(type(output_image[0]))

    image =output_image[0]
    print (image.shape)
    image = (image*255.0).transpose(1,2,0).astype(np.uint8)[:,:,::-1]
    print (image.shape)
    cv2.imwrite('out_optim.png', image)

def main(args):
    benchmark(args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                            help="path of configuration yaml file"),
    parser.add_argument("--model_path", type=str, default="ONNX/",
                            help="path of onnx extra data folder"),
    parser.add_argument("--shape_predictor", type=str, default="preprocess/shape_predictor_68_face_landmarks.dat",
                            help="path of z_id tensor")
    parser.add_argument("--z_id_path", type=str, default="preprocess/z_id.npy",
                            help="path of z_id tensor"),
    parser.add_argument("--target_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000056.png",
                            help="path of preprocessed target face image"),

    args = parser.parse_args()

    main(args)
