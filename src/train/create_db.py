import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from utils import get_meta


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("--train_out", "-o", type=str, required=True,default='./data/imdb_train.mat',
    parser.add_argument("--train_out", "-o", type=str,default='./data/imdb_train.mat',
                        help="path to output database mat file")
    parser.add_argument("--val_out",type=str,default='./data/imdb_val.mat')
    parser.add_argument("--db", type=str, default="imdb",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=224,
                        help="output image size")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_path = args.train_out
    valout_path = args.val_out
    db = args.db
    img_size = args.img_size
    min_score = args.min_score

    root_path = "/data/{}_crop/".format(db)
    mat_path = "./data/{}_crop/".format(db)
    mat_path = mat_path + "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

    out_genders = []
    out_ages = []
    #out_imgs = []
    img_path = []
    val_genders = []
    val_ages = []
    val_path = []
    f_count =0
    m_count =0

    for i in tqdm(range(len(face_score))):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue
        if i < (len(face_score)-10000):
            out_genders.append(int(gender[i]))
            if int(gender[i])==0:
                f_count+=1
            elif int(gender[i])==1:
                m_count+=1
            out_ages.append(age[i])
            #img = cv2.imread(root_path + str(full_path[i][0]))
            img_path.append(root_path + str(full_path[i][0]))
            #print(full_path[i][0])
            #out_imgs.append(cv2.resize(img, (img_size, img_size)))
        else:
            val_genders.append(int(gender[i]))
            val_ages.append(age[i])
            val_path.append(root_path + str(full_path[i][0]))

    output = {"image_path": np.array(img_path), "gender": np.array(out_genders), "age": np.array(out_ages),
              "db": db, "img_size": img_size, "min_score": min_score}
    valout = {"image_path": np.array(val_path), "gender": np.array(val_genders), "age": np.array(val_ages),
              "db": db, "img_size": img_size, "min_score": min_score}
    #scipy.io.savemat(output_path, output)
    #scipy.io.savemat(valout_path, valout)
    print("the femal num ",f_count)
    print("the male  num ",m_count)

if __name__ == '__main__':
    main()
