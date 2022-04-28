from array import array
import onnx
import numpy as np
import os
import cv2
import argparse
import tensorflow as tf
from matplotlib import pyplot as plt
import time
import torch
from torchvision import transforms
from omegaconf import OmegaConf
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader

import torchvision
import pytorch_lightning as pl
from PIL import Image

from aei_net import AEINet

#function that tests tensorflow model of multi level encoder
def test_mle(args, model, device):
    #load target model
    target_img = transforms.ToTensor()(Image.open(args.target_image)).unsqueeze(0).to(device)
    source_img = transforms.ToTensor()(Image.open(args.source_image)).unsqueeze(0).to(device)

    feature_map = model.E(target_img)

    #load and convert model to test
    MultilevelEncoder_onnx = onnx.load(args.model_path + "MultilevelEncoder.onnx")
    tf_model = tf.from_saved_model(args.model_path + "MultilevelEncoder.onnx")
    
    #run inference
    
    feature_map_tf = tf_model.signatures["serving_default"](tf.constant(source_img.cpu().numpy()))

    for i in range(0,7):
        print(feature_map[i].cpu()-feature_map_tf[i])

def main(args):
    #load model
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')

    hp = OmegaConf.load(args.config)
    model = AEINet.load_from_checkpoint(args.checkpoint_path, hp=hp)
    model.eval()
    model.freeze()
    model.to(device)

    test_mle(args, model, device)


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
    parser.add_argument("--target_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000003.png",
                            help="path of preprocessed target face image"),
    parser.add_argument("--source_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000002.png",
                            help="path of preprocessed source face image"),
    parser.add_argument("--gpu_num", type=int, default=0,
                    help="number of gpu"),
    parser.add_argument("--checkpoint_path", type=str, default="chkpt/30.ckpt",
                    help="path of aei-net pre-trained file"),
    parser.add_argument("--output_path", type=str, default="output.jpg",
                    help="path to the output image"),
    parser.add_argument("--output_path_opt", type=str, default="output_opt.jpg",
                    help="path to the output image")

    args = parser.parse_args()

    main(args)