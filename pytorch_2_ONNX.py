import argparse
from tabnanny import verbose
from PIL import Image
from omegaconf import OmegaConf
import torch.onnx
import torch
from torchvision import transforms
import torch.nn.functional as F


from aei_net import AEINet
from dataset import *

def convert_scripting(model, output_path, input):
    script = torch.jit.trace(model, input)
    torch.onnx.export(script, output_path,
                    export_params=True,opset_version=11, verbose=True)

def main(args):
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')

    op = OmegaConf.load(args.config)
    model = AEINet.load_from_checkpoint(args.checkpoint_path, hp=op)

    model.eval()
    model.freeze()
    model.to(device)

    target_img = transforms.ToTensor()(Image.open(args.target_image)).unsqueeze(0).to(device)
    source_img = transforms.ToTensor()(Image.open(args.source_image)).unsqueeze(0).to(device)

    z_id = model.Z(F.interpolate(source_img, size=112, mode='bilinear'))
    z_id = F.normalize(z_id)
    z_id = z_id.detach()

    feature_map = model.E(target_img)
    input_name_MLE = ["target"]
    output_names_MLE = [ "z_%d" % i for i in range(1, 8) ]
    input_names_ADD = [ "z_id" ] + [ "z_%d" % i for i in range(1, 9) ]
    output_names_ADD = [ "output"]
    """
    torch.onnx.export(model.G,(z_id, feature_map), args.output_path + "ADD_gen.onnx",
                    export_params=True,opset_version=11, verbose=True,
                    input_names=input_names_ADD, output_names=output_names_ADD)

        torch.onnx.export(model.E,(target_img), args.output_path + "MultilevelEncoder.onnx", 
                        export_params=True,opset_version=11, verbose=True,  
                        input_names=input_name_MLE, output_names=output_names_MLE)
    """
    #example output
    with torch.no_grad():
        output, z_id, output_z_id, feature_map, output_feature_map = model.forward(target_img, source_img)

    script = torch.jit.trace(model, (target_img, source_img), strict = False)
    script_add = torch.jit.trace(model.G, (z_id, feature_map), strict=False)
    script_mle = torch.jit.trace(model.E, (target_img), strict=False)

    torch.onnx.export(script, (target_img, source_img), args.output_path + "AEI_gen.onnx",
                    export_params=True,opset_version=11, verbose=True
                    , example_outputs=(output, z_id, output_z_id, feature_map, output_feature_map))
    """
        torch.onnx.export(script_mle, target_img, args.output_path + "MultilevelEncoder.onnx",
                    export_params=True,opset_version=11, verbose=True,
                    example_outputs=feature_map)
    
    torch.onnx.export(script_add, (z_id, feature_map), args.output_path + "ADD_gen.onnx",
                    export_params=True,opset_version=11, verbose=True, 
                    example_outputs=output)
    """

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                        help="path of configuration yaml file")
    parser.add_argument("--checkpoint_path", type=str, default="chkpt/30.ckpt",
                        help="path of aei-net pre-trained file")
    parser.add_argument("--output_path", type=str, default="ONNX/single/",
                        help="path of output onnx"),
    parser.add_argument("--target_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000011.png",
                        help="path of preprocessed target face image"),
    parser.add_argument("--source_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000036.png",
                        help="path of preprocessed source face image"),
    parser.add_argument("--output_image", type=str, default="output.png",
                        help="path of output image"),
    parser.add_argument("--gpu_num", type=int, default=0,
                        help="number of gpu")
    args = parser.parse_args()

    main(args)