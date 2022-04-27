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
from torchvision.models import resnet101
import pytorch_lightning as pl
from PIL import Image

from aei_net import AEINet

#function that tests multi level encoder tf lite model
def test_multi_level_encoder(args, model, device):
    #load data
    img = cv2.imread(args.target_image)
    #resize 256
    img = cv2.resize(img, (256, 256))
    #swap rgb to bgr
    img = img.astype(np.float32)
    img = img/255.0

    img = np.transpose(img, (2, 0, 1))

    img = np.expand_dims(img, axis=0)

    interpreter = tflite.Interpreter(args.model_path+ "MultiLevelEncoder_gen_Lite_optimized.tflite", num_threads=12)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()
    z = [interpreter.get_tensor(output_details[1]['index']), interpreter.get_tensor(output_details[0]['index']), interpreter.get_tensor(output_details[3]['index']),
         interpreter.get_tensor(output_details[5]['index']), interpreter.get_tensor(output_details[6]['index']), interpreter.get_tensor(output_details[4]['index']),
         interpreter.get_tensor(output_details[7]['index']), interpreter.get_tensor(output_details[2]['index'])]

    # generate data to compare to

    tmp_img = Image.open(args.target_image)
    target_img = transforms.ToTensor()(Image.open(args.target_image)).unsqueeze(0).to(device)
    tmp_src = Image.open(args.source_image)
    source_img = transforms.ToTensor()(Image.open(args.source_image)).unsqueeze(0).to(device)


    z_id = model.Z(F.interpolate(source_img, size=112, mode='bilinear'))
    z_id = F.normalize(z_id)
    z_id = z_id.detach()
    feature_map = model.E(target_img)

    for i in range(0,7):
        print(feature_map[i].cpu()-z[i])

#function that test add model
def test_add(args, model, device):
    #prepare data
    tmp_img = Image.open(args.target_image)
    target_img = transforms.ToTensor()(Image.open(args.target_image)).unsqueeze(0).to(device)
    tmp_src = Image.open(args.source_image)
    source_img = transforms.ToTensor()(Image.open(args.source_image)).unsqueeze(0).to(device)


    z_id = model.Z(F.interpolate(source_img, size=112, mode='bilinear'))
    z_id = F.normalize(z_id)
    z_id = z_id.detach()
    feature_map = model.E(target_img)

    """ feature map shapes
    torch.Size([1, 1024, 2, 2])
    torch.Size([1, 2048, 4, 4])
    torch.Size([1, 1024, 8, 8])
    torch.Size([1, 512, 16, 16])
    torch.Size([1, 256, 32, 32])
    torch.Size([1, 128, 64, 64])
    torch.Size([1, 64, 128, 128])
    torch.Size([1, 64, 256, 256])
    """

    #load add tflite model
    
    interpreter = tflite.Interpreter(args.model_path+ "ADD_gen_Lite_optimized.tflite", num_threads=12)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    z1 = feature_map[0].detach().cpu().numpy()
    z2 = feature_map[1].detach().cpu().numpy()
    z3 = feature_map[2].detach().cpu().numpy()
    z4 = feature_map[3].detach().cpu().numpy()
    z5 = feature_map[4].detach().cpu().numpy()
    z6 = feature_map[5].detach().cpu().numpy()
    z7 = feature_map[6].detach().cpu().numpy()
    z8 = feature_map[7].detach().cpu().numpy()
    
    print ("zid")
    print (z_id)
    print (z_id.cpu().numpy())

    interpreter.set_tensor(input_details[0]['index'], z5)
    interpreter.set_tensor(input_details[1]['index'], z_id.cpu().detach().numpy())
    interpreter.set_tensor(input_details[2]['index'], z6)
    interpreter.set_tensor(input_details[3]['index'], z2)
    interpreter.set_tensor(input_details[4]['index'], z1)
    interpreter.set_tensor(input_details[5]['index'], z3)
    interpreter.set_tensor(input_details[6]['index'], z7)
    interpreter.set_tensor(input_details[7]['index'], z8)
    interpreter.set_tensor(input_details[8]['index'], z4)
    interpreter.invoke()

    for i in input_details:
        print(i["shape"])

    output_data = interpreter.get_tensor(output_details[0]['index'])

    opt = np.transpose(output_data[0], (1, 2, 0))
    print(opt.min())
    print(opt.max())

#function that compares the optimized model and the original model
def compare_models(args, model, device):
    #load data
    img = cv2.imread(args.target_image)
    #resize 256
    img = cv2.resize(img, (256, 256))
    #swap rgb to bgr
    img = img.astype(np.float32)
    img = img/255.0

    img = np.transpose(img, (2, 0, 1))

    img = np.expand_dims(img, axis=0)

    interpreter = tflite.Interpreter(args.model_path+ "MultiLevelEncoder_gen_Lite_optimized.tflite", num_threads=12)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()
    z = [interpreter.get_tensor(output_details[1]['index']), interpreter.get_tensor(output_details[0]['index']), interpreter.get_tensor(output_details[3]['index']),
         interpreter.get_tensor(output_details[5]['index']), interpreter.get_tensor(output_details[6]['index']), interpreter.get_tensor(output_details[4]['index']),
         interpreter.get_tensor(output_details[7]['index']), interpreter.get_tensor(output_details[2]['index'])]

    # use data in the model to generate the image
    source_img = transforms.ToTensor()(Image.open(args.source_image)).unsqueeze(0).to(device)
    z_id = model.Z(F.interpolate(source_img, size=112, mode='bilinear'))
    z_id = F.normalize(z_id)
    z_id = z_id.detach()
    #run the model
    with torch.no_grad():
        out = model.G(z_id, z)
    out = transforms.ToPILImage()(out.cpu().squeeze().clamp(0, 1))
    
    #load add tflite model
    interpreter = tflite.Interpreter(args.model_path+ "ADD_gen_Lite_optimized.tflite", num_threads=12)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], z[5])
    interpreter.set_tensor(input_details[1]['index'], z_id.cpu().detach().numpy())
    interpreter.set_tensor(input_details[2]['index'], z[6])
    interpreter.set_tensor(input_details[3]['index'], z[2])
    interpreter.set_tensor(input_details[4]['index'], z[1])
    interpreter.set_tensor(input_details[5]['index'], z[3])
    interpreter.set_tensor(input_details[6]['index'], z[7])
    interpreter.set_tensor(input_details[7]['index'], z[8])
    interpreter.set_tensor(input_details[8]['index'], z[4])
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    opt = np.transpose(output_data[0], (1, 2, 0))

    out.show()
    cv2.imshow("opt", opt)
    cv2.waitKey(0)


def main(args):
    #load model
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')

    hp = OmegaConf.load(args.config)
    model = AEINet.load_from_checkpoint(args.checkpoint_path, hp=hp)
    model.eval()
    model.freeze()
    model.to(device)

    compare_models(args, model, device)


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

    args = parser.parse_args()

    main(args)