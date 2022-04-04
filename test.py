import tensorflow as tf
print(tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(
    physical_devices[0], True
)
import numpy as np
    

model=tf.saved_model.load(
    "ONNX/ADD_gen", tags=None, options=None
)

print(model.signatures["serving_default"])

converter = tf.lite.TFLiteConverter.from_saved_model("ONNX/ADD_gen")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset_gen():
    
    for _ in range(100):
        yield {"input.93":     np.random.rand(1, 128, 64, 64).astype(np.float32), 
                "input.5":      np.random.rand(1, 512, 16, 16).astype(np.float32),
                "input.67":     np.random.rand(1, 64, 128, 128).astype(np.float32),
                "input.171":    np.random.rand(1, 1024, 8, 8).astype(np.float32),
                "input.119":    np.random.rand(1, 256, 32, 32).astype(np.float32),
                "input.27":     np.random.rand(1, 256).astype(np.float32),
                "input.145":    np.random.rand(1, 64, 256, 256).astype(np.float32),
                "input.7":      np.random.rand(1, 1024, 2, 2).astype(np.float32),
                "input.47":     np.random.rand(1, 2048, 4, 4).astype(np.float32)}

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

#convert the model
tflite_quant_model = converter.convert()

#save the model
with open("ADD_gen_Lite_optimized.tflite", 'wb') as f:
    f.write(tflite_quant_model)