import argparse
from PIL import Image
from omegaconf import OmegaConf
import torch
from torchvision import transforms
import torch.nn.functional as F

from aei_net import AEINet
from dataset import *




def main(args):
    #load model
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')
    hp = OmegaConf.load(args.config)

    model = AEINet.load_from_checkpoint(args.checkpoint_path, hp=hp)
    model.eval()
    model.freeze()
    model.to(device)

    #load and pre process source image
    source_img = Image.open(args.source_image)
    source_img = transforms.ToTensor()(source_img).unsqueeze(0).to(device)

    with torch.no_grad():
        z_id = model.Z(F.interpolate(source_img, size=112, mode='bilinear'))
        z_id = F.normalize(z_id)
        z_id = z_id.detach()
    #save z_id
    print(type(z_id))
    z_id = z_id.cpu().numpy()
    print(type(z_id))

    np.save(args.save_path, z_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                        help="path of configuration yaml file"),
    parser.add_argument("--checkpoint_path", type=str, default="chkpt/30.ckpt",
                        help="path of aei-net pre-trained file"),
    parser.add_argument("--gpu_num", type=int, default=0,
                        help="number of gpu"),
    parser.add_argument("--source_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000056.png",
                    help="path of preprocessed source face image"),
    parser.add_argument("--save_path", type=str, default="preprocess/z_id.npy",
                        help="path of z_id tensor")

    args = parser.parse_args()

    main(args)