from cgitb import enable
from re import T
import tensorflow as tf
import argparse
from PIL import Image
from omegaconf import OmegaConf
import torch.onnx
import torch
from torchvision import transforms
import torch.nn.functional as F
from aei_net import AEINet
from dataset import *
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                    help="path of configuration yaml file"),
parser.add_argument("--model_path", type=str, default="ONNX/",
                    help="path of onnx extra data folder"),
parser.add_argument("--checkpoint_path", type=str, default="chkpt/30.ckpt",
                    help="path of aei-net pre-trained file"),
parser.add_argument("--images_folder", type=str, default="data/faceshifter-datasets-preprocessed/train/",
                    help="path of preprocessed source face image"),
parser.add_argument("--gpu_num", type=int, default=0,
                    help="number of gpu"),
parser.add_argument("--num_images", type=int, default=4,
                    help="number of images used to convert the model")

args = parser.parse_args()


def optimizeMultiLevelEncoder(argument):
    
    device = torch.device(f"cuda:{argument.gpu_num}" if torch.cuda.is_available() else 'cpu')

    converter = tf.lite.TFLiteConverter.from_saved_model(argument.model_path + "MultilevelEncoder")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        
        for i in range(argument.num_images):
            #choose target image
            target_img_number = (i)
            target_img_path = os.path.join(argument.images_folder, f"{target_img_number:08}.png")
            
            target_img = transforms.ToTensor()(Image.open(target_img_path)).unsqueeze(0)
            yield [(target_img)]
    
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    #convert the model
    tflite_quant_model = converter.convert()
    #save the model
    with open(argument.model_path + "MultiLevelEncoder_gen_Lite_optimized.tflite", 'wb') as f:
        f.write(tflite_quant_model)


def optizeADD(argument):

    device = torch.device(f"cuda:{argument.gpu_num}" if torch.cuda.is_available() else 'cpu')
    
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(
        physical_devices[0], True
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(argument.model_path + "ADD_gen")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    #setup for data preparation
    hp = OmegaConf.load(argument.config)
    model = AEINet.load_from_checkpoint(argument.checkpoint_path, hp=hp)

    model.eval()
    model.freeze()
    model.to(device)

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
            
            target_img = transforms.ToTensor()(Image.open(target_img_path)).unsqueeze(0).to(device)

            feature_map = model.E(target_img)
            item = {"z_id": z_id, "feature_map": feature_map}
            yield {
                "z_id": z_id,
                "z_att": feature_map
            }


    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    #convert the model
    tflite_quant_model = converter.convert()

    #save the model
    with open(args.model_path + "ADD_gen_Lite_optimized.tflite", 'wb') as f:
        f.write(tflite_quant_model)

optizeADD(args)