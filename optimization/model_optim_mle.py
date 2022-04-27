from array import array
import numpy as np
import os
import cv2
import argparse
from tensorflow import lite as tflite
from matplotlib import pyplot as plt
import time
import torch
from torchvision import transforms
from omegaconf import OmegaConf
import torch.nn.functional as F 
import tensorflow as tf

from torch.optim import Adam
from torch.utils.data import DataLoader

import torchvision
import pytorch_lightning as pl
from PIL import Image

from aei_net import AEINet
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="../config/train.yaml",
                    help="path of configuration yaml file"),
parser.add_argument("--model_path", type=str, default="../ONNX/",
                    help="path of onnx extra data folder"),
parser.add_argument("--checkpoint_path", type=str, default="../chkpt/30.ckpt",
                    help="path of aei-net pre-trained file"),
parser.add_argument("--images_folder", type=str, default="../data/faceshifter-datasets-preprocessed/train/",
                    help="path of preprocessed source face image"),
parser.add_argument("--gpu_num", type=int, default=0,
                    help="number of gpu"),
parser.add_argument("--num_images", type=int, default=50,
                    help="number of images used to convert the model")

args = parser.parse_args()


def optizeADD_w_optim_MLE(argument):

    device = torch.device(f"cuda:{argument.gpu_num}" if torch.cuda.is_available() else 'cpu')
    #set experimental memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(
        physical_devices[0], True
    )
    #load model for converter
    converter = tf.lite.TFLiteConverter.from_saved_model(argument.model_path + "ADD_gen")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    #load model for data preparation
    hp = OmegaConf.load(argument.config)
    model = AEINet.load_from_checkpoint(argument.checkpoint_path, hp=hp)

    model.eval()
    model.freeze()
    model.to(device)

    interpreter = tflite.Interpreter(args.model_path+ "MultiLevelEncoder_gen_Lite_optimized.tflite", num_threads=12)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #setup for data preparation
    def representative_dataset_gen():
        
        for i in range(argument.num_images):
            #choose a picture
            source_img_path = os.path.join(argument.images_folder, f"{i:08}.png")
            source_img = transforms.ToTensor()(Image.open(source_img_path)).unsqueeze(0).to(device)

            #prepare the image for the model
            z_id = model.Z(F.interpolate(source_img, size=112, mode='bilinear'))
            z_id = F.normalize(z_id)
            z_id = z_id.detach()

            #choose target image
            target_img_number = (i+argument.num_images)
            target_img_path = os.path.join(argument.images_folder, f"{target_img_number:08}.png")
            img = cv2.imread(target_img_path)
            img = cv2.resize(img, (256, 256))
            
            img = img.astype(np.float32)
            img = img/255.0

            img = np.transpose(img, (2, 0, 1))

            img = np.expand_dims(img, axis=0)

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            feature_map = [interpreter.get_tensor(output_details[1]['index']), interpreter.get_tensor(output_details[0]['index']), interpreter.get_tensor(output_details[3]['index']),
                interpreter.get_tensor(output_details[5]['index']), interpreter.get_tensor(output_details[6]['index']), interpreter.get_tensor(output_details[4]['index']),
                interpreter.get_tensor(output_details[7]['index']), interpreter.get_tensor(output_details[2]['index'])]
            #converting to cpu and numpy and prepraring with dictionary signature
            yield {"input.5": z_id.cpu().numpy(),
                    "input.119": feature_map[5],
                    "input.145": feature_map[6],
                    "input.171": feature_map[7],
                    "input.27": feature_map[1],
                    "input.47": feature_map[2],
                    "input.67": feature_map[3],
                    "input.7": feature_map[0],
                    "input.93": feature_map[4]}

    #converter setup
    converter.representative_dataset = representative_dataset_gen
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #converter.inference_input_type = tf.float32
    #converter.inference_output_type = tf.float32

    #convert the model
    tflite_quant_model = converter.convert()

    #save the model
    with open(args.model_path + "ADD_gen_Lite_optimized.tflite", 'wb') as f:
        f.write(tflite_quant_model)

optizeADD_w_optim_MLE(args)
