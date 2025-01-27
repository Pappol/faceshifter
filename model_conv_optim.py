import argparse
from PIL import Image
from omegaconf import OmegaConf
import torch.onnx
import torch
from torchvision import transforms
import torch.nn.functional as F
import onnx
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

#optimization function for multi-level encoder
def optimizeMultiLevelEncoder(argument):
    
    device = torch.device(f"cuda:{argument.gpu_num}" if torch.cuda.is_available() else 'cpu')

    converter = tf.lite.TFLiteConverter.from_saved_model(argument.model_path + "MultilevelEncoder")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        
        for i in range(argument.num_images):
            #choose target image
            target_img_number = (i)
            target_img_path = os.path.join(argument.images_folder, f"{target_img_number:08}.png")
            
            target_img = transforms.ToTensor()(Image.open(target_img_path)).unsqueeze(0)
            yield [(target_img)]
    
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    #convert the model
    tflite_quant_model = converter.convert()
    #save the model
    with open(argument.model_path + "MultiLevelEncoder_gen_Lite_optimized.tflite", 'wb') as f:
        f.write(tflite_quant_model)

def optizeADD(argument):

    device = torch.device(f"cuda:{argument.gpu_num}" if torch.cuda.is_available() else 'cpu')
    
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(
        physical_devices[0], True
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(argument.model_path + "ADD_gen")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    #setup for data preparation
    hp = OmegaConf.load(argument.config)
    model = AEINet.load_from_checkpoint(argument.checkpoint_path, hp=hp)

    model.eval()
    model.freeze()
    model.to(device)

    def representative_dataset_gen():
        
        for i in range(argument.num_images):
            #choose a picture
            source_img_path = os.path.join(argument.images_folder, f"{i:08}.png")
            source_img = transforms.ToTensor()(Image.open(source_img_path)).unsqueeze(0).to(device)

            #prepare the image for the model
            z_id = model.Z(F.interpolate(source_img, size=112, mode='bilinear'))
            z_id = F.normalize(z_id)
            z_id = z_id.detach()

            #choose target image
            target_img_number = (i+argument.num_images)
            target_img_path = os.path.join(argument.images_folder, f"{target_img_number:08}.png")
            
            target_img = transforms.ToTensor()(Image.open(target_img_path)).unsqueeze(0).to(device)

            feature_map = model.E(target_img)

            yield {'input.5': z_id.cpu().numpy(),
                    "input.119": feature_map[5].cpu().numpy(),
                    "input.145": feature_map[6].cpu().numpy(),
                    "input.171": feature_map[7].cpu().numpy(),
                    "input.27": feature_map[1].cpu().numpy(),
                    "input.47": feature_map[2].cpu().numpy(),
                    "input.67": feature_map[3].cpu().numpy(),
                    "input.7": feature_map[0].cpu().numpy(),
                    "input.93": feature_map[4].cpu().numpy()}

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    #convert the model
    tflite_quant_model = converter.convert()

    #save the model
    with open(args.model_path + "ADD_gen_Lite_optimized.tflite", 'wb') as f:
        f.write(tflite_quant_model)


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