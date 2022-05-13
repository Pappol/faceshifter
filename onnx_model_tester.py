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


#test pytorch model with onnx
def test_py_onnx(args):
    device = 'cpu'

    hp = OmegaConf.load(args.config)
    model = AEINet.load_from_checkpoint(args.checkpoint_path, hp=hp)
    model.eval()
    model.freeze()
    model.to(device)
    # target_img = transforms.ToTensor()(Image.open(args.target_image)).unsqueeze(0).to(device)

    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    ort_session = ort.InferenceSession(args.model_path+"MultilevelEncoder.onnx" , providers = EP_list)
    ort_session_ADD = ort.InferenceSession(args.model_path+"ADD_gen.onnx" , providers = EP_list)

    # target_img = Image.open(args.target_image)
    # target = np.array(target_img).astype(np.float32)
    # target = target.transpose(2,0,1)
    # target = target[np.newaxis, :]

    target=cv2.imread(args.target_image).astype(np.float32)
    target=cv2.resize(target, (256,256))
    target=cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    target = target.transpose(2,0,1)/255.0
    target = target[np.newaxis, :]

    # source_img = transforms.ToTensor()(Image.open(args.source_image)).unsqueeze(0).to(device)
    source_image=cv2.imread(args.source_image)
    source_image=cv2.resize(source_image, (256,256))
    source_image=cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    source_img = transforms.ToTensor()(source_image).unsqueeze(0).to(device)
    

    z_id = model.Z(F.interpolate(source_img, size=112, mode='bilinear'))
    z_id = F.normalize(z_id)
    z_id = z_id.detach()

    outputs = ort_session.run(
        None,
        {"target": target},
    )
    #convert to tensor outputs
    feature_map_torch = []
    feature_map = []
    for i in outputs:
        feature_map_torch.append(torch.from_numpy(i))
        feature_map.append(i)
        print (i.dtype)
        print (i.shape)

    with torch.no_grad():
        image_torch= model.G(z_id, feature_map_torch)

    output = ort_session_ADD.run([], {
                "z_id": z_id.cpu().numpy(), 
                "z_1": feature_map[0], 
                "z_2": feature_map[1], 
                "z_3": feature_map[2], 
                "z_4": feature_map[3], 
                "z_5": feature_map[4], 
                "z_6": feature_map[5], 
                "z_7": feature_map[6], 
                "z_8": feature_map[7]})
    print(np.max(output))
    print(np.min(output))
    # print(output)
    print(type(output[0]))
    image =output[0]
    print (image.shape)
    image = (image[0]*255.0).transpose(1,2,0).astype(np.uint8)[:,:,::-1]
    print (image.shape)
    cv2.imwrite('out_onnx.png', image)
        
    output = transforms.ToPILImage()(torch.squeeze(image_torch.cpu().clamp(0, 1)))
    output.save(args.output_image)



def main(args):

    test_py_onnx(args)


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