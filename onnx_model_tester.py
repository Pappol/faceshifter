from re import M
import onnx
import argparse
import onnxruntime
from onnx import numpy_helper
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch
from torchvision import transforms
import cv2
from aei_net import AEINet


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="ONNX/single/ADD_gen.onnx",
                    help="path of onnx file to convert")
parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                            help="path of configuration yaml file"),
parser.add_argument("--output_path", type=str, default="ONNX/inferred.onnx",
                    help="path of onnx output"),
parser.add_argument("--checkpoint_path", type=str, default="chkpt/30.ckpt",
                    help="path of aei-net pre-trained file"),
parser.add_argument("--target_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000003.png",
                            help="path of preprocessed target face image"),
parser.add_argument("--gpu_num", type=int, default=0,
                    help="number of gpu"),
parser.add_argument("--save_path", type=str, default="image.npy")
args = parser.parse_args()

EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = onnxruntime.InferenceSession(args.model_path, providers=EP_list)
input = session.get_inputs()

#load input data
z_id = np.load("preprocess/z_id.npy").astype(np.float32)
device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')

hp = OmegaConf.load(args.config)
model = AEINet.load_from_checkpoint(args.checkpoint_path, hp=hp)
model.eval()
model.freeze()
model.to(device)
target_img = transforms.ToTensor()(Image.open(args.target_image)).unsqueeze(0).to(device)

feature_map = model.E(target_img)

#run the model
output = session.run([], {input[0].name: z_id, input[1].name: feature_map[0].cpu().detach().numpy(), 
            input[2].name: feature_map[1].cpu().detach().numpy(), input[3].name: feature_map[2].cpu().detach().numpy(), 
            input[4].name: feature_map[3].cpu().detach().numpy(), input[5].name: feature_map[4].cpu().detach().numpy(), 
            input[6].name: feature_map[5].cpu().detach().numpy(), input[7].name: feature_map[6].cpu().detach().numpy(), 
            input[8].name: feature_map[7].cpu().detach().numpy()})
for i in output:
    print (i.shape)
print(type(output[0]))
image =output[0]
print (image.shape)
image = image[0].transpose(1,2,0).astype(np.uint8)
print (image.shape)
np.save(args.save_path, image)

"""
input order
input.5
input.7
input.27
input.47
input.67
input.93
input.119
input.145
input.171

"""