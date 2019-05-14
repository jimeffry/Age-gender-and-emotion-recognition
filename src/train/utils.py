from scipy.io import loadmat, whosmat
from datetime import datetime
import os
import numpy as np
import cv2
import string


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    print(whosmat(mat_path))
    print("path shape ",np.shape(meta[db][0, 0]["full_path"]))
    print("keys ",meta.keys())
    print("2 keys ",meta[db].keys())
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    face_location = meta[db][0,0]["face_location"][0]
    celeb_id = meta[db][0,0]["celeb_id"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_data(mat_path):
    d = loadmat(mat_path)
    print(whosmat(mat_path))
    return d["image_path"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]


def mk_dir(path):
    try:
        #os.mkdir( dir )
        #os.makedirs(path,mode=0755)
        os.makedirs(path)
    except OSError:
        pass

if __name__ == "__main__":
    path = './data/imdb_train.mat'
    image, gender, age, _, image_size, _ = load_data(path)
    print("path ",np.shape(image))
    print("gender ",np.shape(gender))
    print("age ",np.shape(age))
    print('.'+image[0])
    print(len(image))
    print(gender[:10])
    ph = '.'+str(image[0])
    ph=string.strip(ph)
    a=cv2.imread(ph)
    print(a==None)
    print(np.shape(a))
