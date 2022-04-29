from array import array
import numpy as np
import os
import cv2
import argparse
import tflite_runtime.interpreter as tflite
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

def main(args):
    #load model
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')

    hp = OmegaConf.load(args.config)
    model = AEINet.load_from_checkpoint(args.checkpoint_path, hp=hp)
    model.eval()
    model.freeze()
    model.to(device)

    target_img = transforms.ToTensor()(Image.open(args.target_image)).unsqueeze(0).to(device)
    source_img = transforms.ToTensor()(Image.open(args.source_image)).unsqueeze(0).to(device)

    z_id = model.Z(F.interpolate(source_img, size=112, mode='bilinear'))
    z_id = F.normalize(z_id)
    z_id = z_id.detach()
    feature_map = model.E(target_img)

    fm = torch.load(args.feature_map_path)

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
                    help="path to the output image"),
    parser.add_argument("--feature_map_path", type=str, default="feature_map.pth",

    args = parser.parse_args()

    main(args)