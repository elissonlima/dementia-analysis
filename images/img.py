import cv2 
from PIL import Image
import numpy as np
import os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGENS_CORRETAS_PATH = os.path.join(CURRENT_PATH,"correct")
OUT_CORRETAS_PATH = os.path.join(CURRENT_PATH,"processed_correct")
OUT_INCORRETAS_PATH = os.path.join(CURRENT_PATH,"processed_incorrect")
IMAGENS_INCORRETAS_PATH = os.path.join(CURRENT_PATH,"incorrect")

def process_images(in_path, out_path):
    for file_ in os.listdir(in_path):
        
        img = cv2.imread(os.path.join(in_path, file_), 
            cv2.IMREAD_GRAYSCALE)

        # kernel = np.ones((5,5),np.uint8)
        # dilation = cv2.erode(img,kernel,iterations = 1)
        
        blur = cv2.GaussianBlur(img,(3,3),0)
        th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)



        Image.fromarray(th3).save(os.path.join(
            out_path,
            file_
        ))

if __name__ == "__main__":
    process_images(IMAGENS_CORRETAS_PATH, OUT_CORRETAS_PATH)
    process_images(IMAGENS_INCORRETAS_PATH, OUT_INCORRETAS_PATH)
