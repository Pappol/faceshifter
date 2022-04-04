import argparse
from PIL import Image
from omegaconf import OmegaConf
import torch.onnx
import torch
from torchvision import transforms
import torch.nn.functional as F


from aei_net import AEINet
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                    help="path of configuration yaml file")
parser.add_argument("--checkpoint_path", type=str, default="chkpt/30.ckpt",
                    help="path of aei-net pre-trained file")
parser.add_argument("--output_path", type=str, default="ONNX/single/",
                    help="path of output onnx"),
parser.add_argument("--target_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000001.png",
                    help="path of preprocessed target face image")
parser.add_argument("--source_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000002.png",
                    help="path of preprocessed source face image")
parser.add_argument("--gpu_num", type=int, default=0,
                    help="number of gpu")
args = parser.parse_args()

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


torch.onnx.export(model.G,(z_id, feature_map), args.output_path + "ADD_gen.onnx", 
                    export_params=True,opset_version=11)

torch.onnx.export(model.E,(target_img), args.output_path + "MultilevelEncoder.onnx", 
                    export_params=True,opset_version=11)
