'''
# auther : lxy
# time : 2017.1.9 /09:56
#project:
# tool: python2
#version: 0.1
#modify:
#name: center loss
#citations: DEX: Deep EXpectation of apparent age from a single image," ICCV, 2015
            Deep expectation of real and apparent age from a single image without facial landmarks," IJCV, 2016
'''
#############################
import cv2
import numpy as np
import numpy.random as nr
from random import shuffle
import os
from utils import load_data
import string


class BatchLoader(object):

    def __init__(self, file_path, batch_size,img_shape,db='imdb'):
        self.batch_size = batch_size
        self.im_list,self.g_labels,self.a_labels = self.image_dir_processor(file_path,db)
        self.idx = 0
        self.data_num = len(self.im_list)
        self.rnd_list = np.arange(self.data_num)
        shuffle(self.rnd_list)
        self.db = db
        #self.img_temp = cv2.imread(string.strip('.'+self.im_list[0]))
        #self.height,self.width,_= np.shape(self.img_temp)
        self.batch_num = np.ceil(self.data_num / batch_size)
        self.img_shape = img_shape
        #self.labels = tf.one_hot(self.labels,on_value=1,off_value=0,depth=526)
        #print(self.labels)

    def next_batch(self):
        batch_images = []
        batch_a_labels = []
        batch_g_labels = []

        for i in range (self.batch_size):
            if self.idx < self.data_num:
                cur_idx = self.rnd_list[self.idx]
                im_path = self.im_list[cur_idx]
                #im_path = string.strip(im_path)
                im_path = im_path.strip()
                #print(im_path)
                if self.db=="FGNET":
                    rd_path = os.path.join('./data',im_path)
                else:
                    rd_path = '.'+im_path
                image = cv2.imread(rd_path,cv2.IMREAD_GRAYSCALE)
                #image = cv2.imread(os.path.join('.',im_path))
                #print(cur_idx)
                #print(rd_path)
                if image is None:
                    print("pass")
                    continue
                #if len(np.shape(image)) != 3:
                    #continue
                row,col = np.shape(image)[0],np.shape(image)[1]
                #print(np.shape(image))
                '''
                image = cv2.imread('.'+im_path)
                row,col,chal = np.shape(image)                
                if  chal == 3:
                    temp_im = np.zeros([row,col,3],dtype=np.float32)
                    temp_im[:,:,0]=image
                    temp_im[:,:,1]=image
                    temp_im[:,:,2]=image
                    image = temp_im
                    '''
                if row != self.img_shape or col !=self.img_shape :
                    image = cv2.resize(image,(self.img_shape,self.img_shape))
                image = np.expand_dims(image,axis=-1)
                batch_images.append(image)
                batch_a_labels.append(self.a_labels[cur_idx])
                batch_g_labels.append(self.g_labels[cur_idx])
                self.idx +=1
            elif self.data_num % self.batch_size !=0:
                remainder = self.data_num % self.batch_size
                patch_num = self.batch_size - remainder
                for j in range(patch_num):
                    cur_idx = self.rnd_list[j]
                    im_path = self.im_list[cur_idx]
                    #im_path = string.strip(im_path)
                    im_path = im_path.strip()
                    #image = cv2.imread(os.path.join('.',im_path))
                    if self.db=="FGNET":
                        rd_path = os.path.join('./data',im_path)
                    else:
                        rd_path = '.'+im_path
                    image = cv2.imread(rd_path,cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                    #if len(np.shape(image)) != 3:
                        #continue
                    #image = np.expand_dims(image,axis=-1)
                    row,col = np.shape(image)[0],np.shape(image)[1]
                    '''
                    image = cv2.imread('.'+im_path)
                    row,col,chal = np.shape(image)
                    if not chal == 3:
                        temp_im = np.zeros([row,col,3],dtype=np.float32)
                        temp_im[:,:,0]=image
                        temp_im[:,:,1]=image
                        temp_im[:,:,2]=image
                        image = temp_im
                    '''
                    if row != self.img_shape or col !=self.img_shape :
                        image = cv2.resize(image,(self.img_shape,self.img_shape))
                    image = np.expand_dims(image,axis=-1)
                    batch_images.append(image)
                    batch_a_labels.append(self.a_labels[cur_idx])
                    batch_g_labels.append(self.g_labels[cur_idx])
                self.idx = 0
                shuffle(self.rnd_list)
                break
            else:
                self.idx = 0
                shuffle(self.rnd_list)
        #print("indx",self.idx,"batch",len(batch_images))
        batch_images = np.array(batch_images).astype(np.float32)
        batch_a_labels = np.array(batch_a_labels).astype(np.int32)
        batch_g_labels = np.array(batch_g_labels).astype(np.int32)
        return batch_images, batch_g_labels,batch_a_labels

    def image_dir_processor(self, file_path,db):
        if not os.path.exists(file_path):
            print ("File %s not exists." % file_path)
            exit()
        if db == "FGNET":
            f= open(file_path,'rb')
            image_path_list = []
            gender = []
            age = []
            for img_path in f.readlines():
                f_list = img_path.split(',')
                #print(img_path)
                image_path_list.append(f_list[0])
                age.append(f_list[1])
                gender.append(0)
        else :
            image_path_list, gender, age, _, image_size, _ = load_data(file_path)

        return image_path_list, gender, age

if __name__ =='__main__':
    data = BatchLoader("./data/imdb_val.mat", 16,64,"imdb")
    a,b,c = data.next_batch()
    print(a.shape)
