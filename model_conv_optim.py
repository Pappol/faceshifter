import argparse
from PIL import Image
from omegaconf import OmegaConf
import torch.onnx
import torch
from torchvision import transforms
import torch.nn.functional as F
import onnx
import argparse
from onnx_tf.backend import prepare
import tensorflow as tf

from aei_net import AEINet
from dataset import *

#function to load model
def load_model(config, checkpoint_path, gpu_num):
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else 'cpu')
    hp = OmegaConf.load(config)
    model = AEINet.load_from_checkpoint(checkpoint_path, hp=hp)
    model.eval()
    model.freeze()
    model.to(device)
    return model, device

#function to load images for conversion
def load_images_from_path(target_image_path, source_img_path, device):
    target_img = transforms.ToTensor()(Image.open(target_image_path)).unsqueeze(0).to(device)
    source_img = transforms.ToTensor()(Image.open(source_img_path)).unsqueeze(0).to(device)
    return target_img, source_img

#function to convert model to ONNX
def pytorch_2_onnx(model, target_img, source_img, output_path, opset_version):
    z_id = model.Z(F.interpolate(source_img, size=112, mode='bilinear'))
    z_id = F.normalize(z_id)
    z_id = z_id.detach()
    feature_map = model.E(target_img)
    torch.onnx.export(model.G,(z_id, feature_map), output_path + "ADD_gen.onnx", 
                        export_params=True,opset_version=opset_version)
    torch.onnx.export(model.E,(target_img), output_path + "MultilevelEncoder.onnx", 
                        export_params=True,opset_version=opset_version)

#function that converts onnx to tf
def convert_to_tf(onnx_model, output_path):
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(output_path)

def representative_dataset_gen_Multi(device, num_images, image_path):
        for i in range(num_images):
            #choose target image
            target_img_number = (i)
            target_img_path = os.path.join(image_path, f"{target_img_number.value:08}.png")
            
            target_img = transforms.ToTensor()(Image.open(target_img_path)).unsqueeze(0)
            yield [(target_img)]


#main function to run the conversion and the optimization
def main(config, checkpoint_path, output_path, target_image, source_image, gpu_num,tflite_path, num_images, image_path):
    #load model
    model, device = load_model(config, checkpoint_path, gpu_num)

    #load images
    target_img, source_img = load_images_from_path(target_image, source_image, device)

    #convert model to ONNX
    pytorch_2_onnx(model, target_img, source_img, output_path, opset_version=11)

    #convert models to TF

    #convert ADD_gen to TF
    ADD_gen_onnx = onnx.load(output_path + "ADD_gen.onnx")
    convert_to_tf(ADD_gen_onnx, tflite_path+ "ADD_tf")


    #convert MultilevelEncoder to TF
    MultilevelEncoder_onnx = onnx.load(output_path + "MultilevelEncoder.onnx")
    convert_to_tf(MultilevelEncoder_onnx, tflite_path+ "MultilevelEncoder_tf")

    #convert to TFLite
    converter_ADD = tf.lite.TFLiteConverter.from_saved_model(tflite_path+ "ADD_tf")
    tflite_ADD = converter_ADD.convert()

    converter_Multi = tf.lite.TFLiteConverter.from_saved_model(tflite_path+ "MultilevelEncoder_tf")
    tflite_Multi = converter_Multi.convert()

    """

    #optimize TFLite
    tflite_ADD.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_Multi.optimizations = [tf.lite.Optimize.DEFAULT]

    #mutli-level encoder
    tflite_Multi.representative_dataset = representative_dataset_gen_Multi(device, num_images, image_path)
    tflite_Multi.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_Multi.inference_input_type = tf.int8
    tflite_Multi.inference_output_type = tf.int8

    tflite_quant_model = tflite_Multi.convert()
    #save optimized TFLite model
    with open(tflite_path + "MultilevelEncoder_quant.tflite", "wb") as f:
        f.write(tflite_quant_model)
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                        help="path of configuration yaml file")
    parser.add_argument("--checkpoint_path", type=str, default="chkpt/30.ckpt",
                        help="path of aei-net pre-trained file")
    parser.add_argument("--onnx_out_path", type=str, default="ONNX/single/",
                        help="path of output onnx"),
    parser.add_argument("--tf_out_path", type=str, default="TF/single/",
                        help="path of output tf"),
    parser.add_argument("--target_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000001.png",
                        help="path of preprocessed target face image"),
    parser.add_argument("--source_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000002.png",
                        help="path of preprocessed source face image"),
    parser.add_argument("--gpu_num", type=int, default=0,
                        help="number of gpu"),
    parser.add_argument("--num_images", type=int, default=100,
                    help="number of images used to convert the model"),
    parser.add_argument("--images_folder", type=str, default="data/faceshifter-datasets-preprocessed/train/",
                    help="path of preprocessed source face image"),
    args = parser.parse_args()

    main(args.config, args.checkpoint_path, args.onnx_out_path, 
        args.target_image, args.source_image, args.gpu_num, args.tf_out_path, args.num_images, args.images_folder)