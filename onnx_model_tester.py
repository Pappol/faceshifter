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
import onnxruntime as ort
import torch.nn.functional as F

def old_test(args):
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

def main(args):

    test_py_onnx(args)


#test pytorch model with onnx
def test_py_onnx(args):
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')

    hp = OmegaConf.load(args.config)
    model = AEINet.load_from_checkpoint(args.checkpoint_path, hp=hp)
    model.eval()
    model.freeze()
    model.to(device)
    target_img = transforms.ToTensor()(Image.open(args.target_image)).unsqueeze(0).to(device)

    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    ort_session = ort.InferenceSession(args.model_path+"MultilevelEncoder.onnx" , providers = EP_list)

    target_img = Image.open(args.target_image)
    target = np.array(target_img).astype(np.float32)
    target = target.transpose(2,0,1)
    target = target[np.newaxis, :]
    source_img = transforms.ToTensor()(Image.open(args.source_image)).unsqueeze(0).to(device)

    z_id = model.Z(F.interpolate(source_img, size=112, mode='bilinear'))
    z_id = F.normalize(z_id)
    z_id = z_id.detach()

    outputs = ort_session.run(
        None,
        {"target": target},
    )
    #convert to tensor outputs
    feature_map = []
    for i in outputs:
        feature_map.append(torch.from_numpy(i))

    for i in feature_map:
        i.to(device)
        print (i.type())
        print (i.shape)

    with torch.no_grad():
        image= model.G(z_id, feature_map)
        
    output = transforms.ToPILImage()(image.cpu().clamp(0, 1))
    output.save(args.output_image)


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ONNX/single/",
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
    parser.add_argument("--output_image", type=str, default="output.png",
                        help="path of output image"),
    parser.add_argument("--source_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000056.png",
                    help="path of preprocessed source face image"),
    parser.add_argument("--save_path", type=str, default="image.npy")
    args = parser.parse_args()

    main(args)