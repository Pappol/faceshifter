import cv2
import numpy as np

#load np array
def load_np_array():
    return np.load("./image.npy")

#main function displays np array as image
def main():
    image = load_np_array()


    cv2.imshow("image", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()