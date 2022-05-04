import onnxruntime as ort
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = ort.InferenceSession("ONNX/single/MultilevelEncoder.onnx" , providers = EP_list)

target_img = Image.open("data/faceshifter-datasets-preprocessed/train/00000011.png")
target = np.array(target_img).astype(np.float32)
target = target.transpose(2,0,1)
target = target[np.newaxis, :]

z_id = np.load("preprocess/z_id.npy").astype(np.float32)


outputs = ort_session.run(
    None,
    {"target": target},
)
for i in outputs:
    print (i.shape)

"""
(1, 1024, 2, 2)
(1, 2048, 4, 4)
(1, 1024, 8, 8)
(1, 512, 16, 16)
(1, 256, 32, 32)
(1, 128, 64, 64)
(1, 64, 128, 128)
(1, 64, 256, 256)
"""
ort_session = ort.InferenceSession("ONNX/single/ADD_gen.onnx" , providers = EP_list)


output_add = ort_session.run(
    None,
    {"z_id": z_id,
    "z_1": outputs[0],
    "z_2": outputs[1],
    "z_3": outputs[2],
    "z_4": outputs[3],
    "z_5": outputs[4],
    "z_6": outputs[5],
    "z_7": outputs[6],
    "z_8": outputs[7]
    }
)
print(output_add[0])

output = transforms.ToPILImage()(output_add[0].squeeze().transpose(1,2,0).astype(np.uint8))
output.save("inferred.png")
