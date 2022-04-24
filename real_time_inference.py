import argparse
from PIL import Image
from omegaconf import OmegaConf
import tflite_runtime.interpreter as tflite
from imutils import face_utils
import dlib
import time
import cv2
import os
import PIL
import random
import numpy as np
import scipy.ndimage
from PIL import Image

def lendmarks(image, detector, shape_predictor):
    output_size = 256
    transform_size=4096
    enable_padding=True
    dets = detector(image, 1)
    if len(dets) <= 0:
        print("no face landmark detected")
        return image
    else:
        shape = shape_predictor(image, dets[0])
        points = np.empty([68, 2], dtype=int)
        for b in range(68):
            points[b, 0] = shape.part(b).x
            points[b, 1] = shape.part(b).y
        lm = points
        # lm = fa.get_landmarks(input_img)[-1]
        # lm = np.array(item['in_the_wild']['face_landmarks'])
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2
        
        image = Image.fromarray(np.uint8(image))
        image = image.convert('RGB')

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(image.size[0]) / shrink)), int(np.rint(float(image.size[1]) / shrink)))
            image = image.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, image.size[0]), min(crop[3] + border, image.size[1]))
        if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
            image = image.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - image.size[0] + border, 0), max(pad[3] - image.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            image = np.pad(np.float32(image), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = image.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            image += (scipy.ndimage.gaussian_filter(image, [blur, blur, 0]) - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            image += (np.median(image, axis=(0,1)) - image) * np.clip(mask, 0.0, 1.0)
        # Transform.
        image = image.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            image = image.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        return image



def main(args):
    
    cap = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    # allow the camera to warmup
    time.sleep(0.1)

    while True:
        ret, frame = cap.read()
        landmarks = lendmarks(frame, detector, predictor)
        
        cv2.imshow('frame', landmarks)
        if cv2.waitKey(20) & 0xFF == 27:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                        help="path of configuration yaml file"),
    parser.add_argument("--model_path", type=str, default="ONNX/",
                        help="path of onnx extra data folder"),
    parser.add_argument("--shape_predictor", type=str, default="preprocess/shape_predictor_68_face_landmarks.dat",
                        help="path of z_id tensor")
    parser.add_argument("--z_id_path", type=str, default="preprocess/z_id.npy",
                        help="path of z_id tensor")

    args = parser.parse_args()

    main(args)