import os
import cv2
import numpy as np

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(CURRENT_PATH)
CORRECT_IMG_PATH = os.path.join(PROJECT_ROOT_PATH, "images", "processed_correct")
INCORRECT_IMG_PATH  = os.path.join(PROJECT_ROOT_PATH, "images", "processed_incorrect")

def get_databases():

    # Mean of shape in this database: 248,320
    database = np.array([])
    labels = np.array([])
    qtd = 0

    if not ( os.path.isfile(os.path.join(PROJECT_ROOT_PATH, "images", "database.npy"))
            or os.path.isfile(os.path.join(PROJECT_ROOT_PATH, "images", "labels.npy")) ):

        for file_ in os.listdir(CORRECT_IMG_PATH):
            im = cv2.imread(os.path.join(CORRECT_IMG_PATH, file_),
                            cv2.IMREAD_GRAYSCALE)
            im_resized = cv2.resize(im,(320, 248))
            im_resized = im_resized.reshape((248,320,1))
            database = np.append(database, im_resized)
            labels = np.append(labels, [1])
            qtd+=1

        for file_ in os.listdir(INCORRECT_IMG_PATH):
            im = cv2.imread(os.path.join(INCORRECT_IMG_PATH, file_),
                            cv2.IMREAD_GRAYSCALE)
            im_resized = cv2.resize(im, (320, 248))
            im_resized = im_resized.reshape((248,320,1))
            database = np.append(database, im_resized)
            labels = np.append(labels, [0])
            qtd+=1

        database = database.reshape((qtd,248,320,1))
        labels = labels.reshape((qtd,1))

        np.save(os.path.join(PROJECT_ROOT_PATH, "images", "database.npy"),
                database)
        np.save(os.path.join(PROJECT_ROOT_PATH, "images", "labels.npy"),
                labels)
    else:
        database = np.load(os.path.join(PROJECT_ROOT_PATH, "images", "database.npy"))
        labels = np.load(os.path.join(PROJECT_ROOT_PATH, "images", "labels.npy"))

    return database, labels