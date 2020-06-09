import cv2 
from PIL import Image
import numpy as np
import os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGENS_CORRETAS_PATH = os.path.join(CURRENT_PATH,"imagens corretas")
OUT_CORRETAS_PATH = os.path.join(CURRENT_PATH,"out_corretas")
IMAGENS_INCORRETAS_PATH = os.path.join(CURRENT_PATH,"imagens incorretas")

if __name__ == "__main__":
    
    for file_ in os.listdir(IMAGENS_CORRETAS_PATH):
        
        img = cv2.imread(os.path.join(IMAGENS_CORRETAS_PATH, file_), 
            cv2.IMREAD_GRAYSCALE)

        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.erode(img,kernel,iterations = 1)
        
        blur = cv2.GaussianBlur(dilation,(7,7),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        Image.fromarray(th3).save(os.path.join(
            OUT_CORRETAS_PATH,
            file_
        ))
