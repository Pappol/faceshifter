import argparse
from PIL import Image
from omegaconf import OmegaConf
import torch.onnx
import torch
from torchvision import transforms

from aei_net import AEINet

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                    help="path of configuration yaml file")
parser.add_argument("--checkpoint_path", type=str, required=True,
                    help="path of aei-net pre-trained file")
parser.add_argument("--target_image", type=str, required=True,
                    help="path of preprocessed target face image")
parser.add_argument("--source_image", type=str, required=True,
                    help="path of preprocessed source face image")
parser.add_argument("--output_path", type=str, default="ONNX/model.onnx",
                    help="path of output onnx")
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


torch.onnx.export(model, (target_img, source_img), args.output_path, 
                    export_params=True,opset_version=11, do_constant_folding=True,
                    input_names = ['input'],
                    use_external_data_format=True,
                    output_names = ['output'],
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
