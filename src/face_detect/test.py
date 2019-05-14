# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/06/12 14:09
#project: Face detect
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face detect testing caffe model
####################################################
import sys
from tqdm import tqdm
sys.path.append('.')
import os
os.environ['GLOG_minloglevel'] = '2'
import cv2
import numpy as np
import argparse
import time
import mxnet as mx
from scipy.spatial import distance
from mtcnn_config import config
#from Detector_caffe import FaceDetector_Opencv,MTCNNDet
from Detector_mxnet import MtcnnDetector 
import shutil
from align import alignImg_opencv,Align_img,alignImg_angle,alignImg
sys.path.append(os.path.join(os.path.dirname(__file__),'../face_recognize'))
from face_model import Img_Pad

def args():
    parser = argparse.ArgumentParser(description="mtcnn caffe")
    parser.add_argument('--file-in',type=str,dest='file_in',default=None,\
                        help="the file input path")
    parser.add_argument('--min-size',type=int,dest='min_size',default=24,\
                        help="scale img size")
    parser.add_argument('--img-path1',type=str,dest='img_path1',default="test1.jpg",\
                        help="img1 saved path")
    parser.add_argument('--img-path2',type=str,dest='img_path2',default="test2.jpg",\
                        help="scale img size")
    parser.add_argument('--base-dir',type=str,dest='base_dir',default="./",\
                        help="images saved dir")
    parser.add_argument('--prototxt',type=str,dest='p_path',default="../models/deploy.prototxt",\
                        help="caffe prototxt path")
    parser.add_argument('--caffemodel',type=str,dest='m_path',default="../models/deploy.caffemodel",\
                        help="caffe model path")
    parser.add_argument('--save-dir',type=str,dest='save_dir',default="./",\
                        help="images saved dir")
    parser.add_argument('--base-name',type=str,dest='base_name',default="videox",\
                        help="images saved dir")
    parser.add_argument('--cmd-type',type=str,dest='cmd_type',default="video",\
                        help="detect face from : video or txtfile")
    parser.add_argument('--save-dir2',type=str,dest='save_dir2',default=None,\
                        help="images saved dir")
    parser.add_argument('--img-size',type=str,dest='img_size',default='112,96',\
                        help="images saved size")
    return parser.parse_args()


def evalu_img(imgpath,min_size):
    #cv2.namedWindow("test")
    #cv2.moveWindow("test",1400,10)
    threshold = np.array([0.5,0.7,0.8])
    base_name = "test_img"
    save_dir = './output'
    crop_size = [112,112]
    #detect_model = MTCNNDet(min_size,threshold)
    detect_model = MtcnnDetector(min_size,threshold)
    #alignface = Align_img(crop_size)
    img = cv2.imread(imgpath)
    h,w = img.shape[:2]
    if config.img_downsample and h > 1000:
        img = img_ratio(img,720)
    rectangles = detect_model.detectFace(img)
    #draw = img.copy()
    if len(rectangles)>0:
        points = np.array(rectangles)
        #print('rec shape',points.shape)
        points = points[:,5:]
        #print("landmarks: ",points)
        points_list = points.tolist()
        crop_imgs = alignImg(img,crop_size,points_list)
        #crop_imgs = alignImg_opencv(img,crop_size,points_list)
        #crop_imgs = alignface.extract_image_chips(img,points_list)
        #crop_imgs = alignImg_angle(img,crop_size,points_list)
        for idx_cnt,img_out in enumerate(crop_imgs):
            savepath = os.path.join(save_dir,base_name+'_'+str(idx_cnt)+".jpg")
            #img_out = cv2.resize(img_out,(96,112))
            #cv2.imshow("test",img_out)
            cv2.imwrite(savepath,img_out)
        for rectangle in rectangles:
            score_label = str("{:.2f}".format(rectangle[4]))
            cv2.putText(img,score_label,(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            cv2.rectangle(img,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
            if len(rectangle) > 5:
                if config.x_y:
                    for i in range(5,15,2):
                        cv2.circle(img,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
                else:
                    rectangle = rectangle[5:]
                    for i in range(5):
                        cv2.circle(img,(int(rectangle[i]),int(rectangle[i+5])),2,(0,255,0))
    else:
        print("No face detected")
    #cv2.imshow("test",img)
    #cv2.waitKey(0)
    #cv2.imwrite('test.jpg',draw)

def show_formtxt(txt_file,min_size):
    f_r = open(txt_file)
    lines = f_r.readlines()
    for line_one in lines:
        evalu_img(line_one.strip(),min_size)

def main():
    cv2.namedWindow("test")
    cv2.moveWindow("test",1400,10)
    threshold = [0.99,0.99,0.9]
    imgpath = "test2.jpg"
    parm = args()
    min_size = parm.min_size
    file_in = parm.file_in
    detect_model = MTCNNDet(min_size,threshold)
    if file_in is None:
        cap = cv2.VideoCapture(0)
        print("read camera")
    else:
        cap = cv2.VideoCapture(file_in)
    if not cap.isOpened():
        print("failed open camera")
        return 0
    else: 
        while 1:
            _,frame = cap.read()
            rectangles = detect_model.detectFace(frame)
            draw = frame.copy()
            if len(rectangles) > 0:
                for rectangle in rectangles:
                    score_label = str("{:.2f}".format(rectangle[4]))
                    cv2.putText(draw,score_label,(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
                    cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
                    if len(rectangle) > 5:
                        if config.x_y:
                            for i in range(5,15,2):
                                cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
                        else:
                            rectangle = rectangle[5:]
                            for i in range(5):
                                cv2.circle(draw,(int(rectangle[i]),int(rectangle[i+5])),2,(0,255,0))
            cv2.imshow("test",draw)
            q=cv2.waitKey(10) & 0xFF
            if q == 27 or q ==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def img_ratio(img,img_n):
    h,w,c = img.shape
    if h > w:
        img_h = img_n
        ratio_ = float(h) / float(w)
        img_w = img_h / ratio_
    else:
        img_w = img_n
        ratio_ = float(h) / float(w)
        img_h = img_w * ratio_
    img_out = cv2.resize(img,(int(img_w),int(img_h)))
    return img_out

def img_crop(img,bbox):
    imgh,imgw,imgc = img.shape
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    if config.id_box_widen:
        boxw = x2-x1
        boxh = y2-y1
        x1 = max(0,int(x1-0.2*boxw))
        y1 = max(0,int(y1-0.1*boxh))
        x2 = min(imgw,int(x2+0.2*boxw))
        #y2 = min(imgh,int(y2+0.1*boxh))
    cropimg = img[y1:y2,x1:x2,:]
    return cropimg

def sort_box(boxes_or):
    boxes = np.array(boxes_or)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = area.argsort()[::-1]
    #print(I)
    #print(boxes_or[0])
    idx = map(int,I)
    return boxes[idx[:]].tolist()

def label_show(img,rectangles):
    '''
    x1,y1,x2,y2,score = rectangles[:5]
    x_y=0, coordinates of the 5 points(eye,nose,mouse) will be ordered x1,x2,x3,x4,x5,y1,y2,y3,y4,y5
    x_y=1, coordinates of the 5 points(eye,nose,mouse) will be ordered x1,y1,x2,y2,...,x5,y5
    '''
    for rectangle in rectangles:
        score_label = str("{:.2f}".format(rectangle[4]))
        cv2.putText(img,score_label,(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        cv2.rectangle(img,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
        if len(rectangle) > 5:
            if config.x_y:
                for i in range(5,15,2):
                    cv2.circle(img,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
            else:
                rectangle = rectangle[5:]
                for i in range(5):
                    cv2.circle(img,(int(rectangle[i]),int(rectangle[i+5])),2,(0,255,0))

def check_boxes(boxes,threshold):
    '''
    x1,y1,x2,y2 = boxes[:4]
    '''
    boxes = np.array(boxes)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    idx = np.where((y2-y1)>threshold)
    #print('idx',idx,len(idx[0]))
    if len(idx[0]) >0:
        return boxes[idx[:]].tolist()
    else:
        return None

def blurness(face):
    score = cv2.Laplacian(face, cv2.CV_64F).var()
    return score

def save_cropfromtxt(file_in,base_dir,save_dir,crop_size,name):
    '''
    file_in: images path recorded
    base_dir: images locate in 
    save_dir: detect faces saved in
    fun: saved id images, save name is the same input image
    '''
    f_ = open(file_in,'r')
    f_out = './output/%s_face.txt' % name
    failed_w = open(f_out,'w')
    lines_ = f_.readlines()
    min_size = 50 #15
    threshold = np.array([0.8,0.8,0.9])
    #detect_model = MTCNNDet(min_size,threshold) 
    detect_model = MtcnnDetector(min_size,threshold)
    #model_path = "../models/haarcascade_frontalface_default.xml"
    #detect_model = FaceDetector_Opencv(model_path)
    if os.path.exists(save_dir):
        pass
    else:
        os.makedirs(save_dir)
    idx_cnt = 0 
    sizefailed_cnt = 0
    blurfailed_cnt = 0
    if config.show:
        cv2.namedWindow("src")
        cv2.namedWindow("crop")
        cv2.moveWindow("crop",650,10)
        cv2.moveWindow("src",100,10)
    total_item = len(lines_)
    for i in tqdm(range(total_item)):
        line_1 = lines_[i]
        line_1 = line_1.strip()
        img_path = os.path.join(base_dir,line_1)
        img = cv2.imread(img_path)
        if img is None:
            print("img is none,",img_path)
            continue
        h,w = img.shape[:2]
        if config.img_downsample and max(w,h) > 1080:
            img = img_ratio(img,640)
        line_s = line_1.split("/")  
        img_name = line_s[-1]
        new_dir = '/'.join(line_s[:-1]) 
        rectangles = detect_model.detectFace(img)
        if len(rectangles)> 0:
            idx_cnt+=1
            rectangles = sort_box(rectangles)
            #rectangles = check_boxes(rectangles,30)
            if rectangles is None:
                failed_w.write(img_path)
                failed_w.write('\n')
                sizefailed_cnt+=1
                continue
            #print("box",np.shape(rectangles))
            if not config.crop_org:
                points = np.array(rectangles)
                points = points[:,5:]
                points_list = points.tolist()
                points_list = [points_list[0]]
                img_out = alignImg(img,crop_size,points_list)
                #img_out = alignImg_opencv(img,crop_size,points_list)
                #img_out = alignImg_angle(img,crop_size,points_list)
                img_out = img_out[0]
            else:
                img_out = img_crop(img,rectangles[0])
                #savepath = os.path.join(save_dir,str(idx_cnt)+".jpg")
                if config.imgpad:
                    img_out = Img_Pad(img_out,crop_size)
                else:
                    img_out = cv2.resize(img_out,(crop_size[1],crop_size[0]))
            '''
            face_score = blurness(img_out)
            if face_score < 100:
                failed_w.write(img_path)
                failed_w.write('\n')
                blurfailed_cnt+=1
                continue
            '''
            line_1 = line_1.split('.')[0] + '.png'
            savepath = os.path.join(save_dir,line_1)
            cv2.imwrite(savepath,img_out)
            cv2.waitKey(10)
            #cv2.imwrite(savepath,img)
            if config.show:
                label_show(img,rectangles)
                cv2.imshow("crop",img_out)
                cv2.waitKey(100)
        else:
            failed_w.write(img_path)
            failed_w.write('\n')
            print("failed ",img_path)
        if config.show:
            cv2.imshow("src",img)
            cv2.waitKey(10)
    failed_w.close()
    f_.close()
    print("size less 112: ",sizefailed_cnt)
    print("blur less 100: ",blurfailed_cnt)


def save_cropfromvideo(file_in,base_name,save_dir,save_dir2,crop_size):
    '''
    file_in: input video file path
    base_name: saved images prefix name
    save_dir: saved images path
    fun: saved detect faces to dir
    '''
    if file_in is None:
        v_cap = cv2.VideoCapture(0)
    else:
        v_cap = cv2.VideoCapture(file_in)
    min_size = 50
    threshold = np.array([0.5,0.8,0.9])
    detect_model = MTCNNDet(min_size,threshold) 
    #model_path = "../models/haarcascade_frontalface_default.xml"
    #detect_model = FaceDetector_Opencv(model_path)
    #crop_size = [112,96]
    #Align_Image = Align_img(crop_size)
    def mk_dirs(path):
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
    def img_crop(img,bbox,w,h):
        x1 = int(max(bbox[0],0))
        y1 = int(max(bbox[1],0))
        x2 = int(min(bbox[2],w))
        y2 = int(min(bbox[3],h))
        cropimg = img[y1:y2,x1:x2,:]
        return cropimg
    def img_crop2(img,bbox,imgw,imgh):
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        if config.box_widen:
            boxw = x2-x1
            boxh = y2-y1
            x1 = int(max(0,int(x1-0.2*boxw)))
            y1 = int(max(0,int(y1-0.1*boxh)))
            x2 = int(min(imgw,int(x2+0.2*boxw)))
            y2 = int(min(imgh,int(y2+0.1*boxh)))
        cropimg = img[y1:y2,x1:x2,:]
        return cropimg
    idx_cnt = 0 
    mk_dirs(save_dir)
    if save_dir2 is not None:
        mk_dirs(save_dir2)
    if not v_cap.isOpened():
        print("field to open video")
    else:
        total_num = v_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fram_w = v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fram_h = v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print("video frame num: ",total_num)
        frame_cnt = 0
        while v_cap.isOpened():
            ret,frame = v_cap.read()
            frame_cnt+=1
            sys.stdout.write('\r>> deal with %d/%d' % (frame_cnt,total_num))
            sys.stdout.flush()
            if ret: 
                #frame = img_ratio(frame,640)
                rectangles = detect_model.detectFace(frame)
            else:
                continue
            if len(rectangles)> 0:
                #rectangles = sort_box(rectangles)
                if config.crop_org:
                    for bbox_one in rectangles:
                        idx_cnt+=1
                        img_out = img_crop(frame,bbox_one,fram_w,fram_h)
                        savepath = os.path.join(save_dir,base_name+'_'+str(idx_cnt)+".png")
                        #savepath = os.path.join(save_dir,line_1)
                        img_out = cv2.resize(img_out,(crop_size[1],crop_size[0]))
                        cv2.imwrite(savepath,img_out)
                else:
                    points = np.array(rectangles)
                    points = points[:,5:]
                    points_list = points.tolist()
                    #crop_imgs = Align_Image.extract_image_chips(frame,points_list)
                    #crop_imgs = alignImg(frame,crop_size,points_list)
                    crop_imgs = alignImg_opencv(frame,crop_size,points_list)
                    for box_idx,img_out in enumerate(crop_imgs):
                        idx_cnt+=1
                        #savepath = os.path.join(save_dir,base_name+'_'+str(idx_cnt)+".jpg")
                        f_name = 'f'+str(frame_cnt)
                        savepath = os.path.join(save_dir,f_name+'_'+str(box_idx)+".png")
                        #img_out = cv2.resize(img_out,(96,112))
                        #cv2.imshow("test",img_out)
                        cv2.imwrite(savepath,img_out)
                        cv2.waitKey(10)
                        if config.box_widen:
                            savepath2 = os.path.join(save_dir2,f_name+'_'+str(idx_cnt)+".png")
                            img_widen = img_crop2(frame,rectangles[box_idx],fram_w,fram_h)
                            cv2.imwrite(savepath2,img_widen)
                            cv2.waitKey(10)
                        #print("crop num,",idx_cnt)
                '''
                savedir = os.path.join(save_dir,new_dir)
                if os.path.exists(savedir):
                    savepath = os.path.join(savedir,img_name)
                    shutil.copyfile(img_path,savepath)
                else:
                    os.makedirs(savedir)
                    savepath = os.path.join(savedir,img_name)
                    shutil.copyfile(img_path,savepath)
                '''
                #cv2.imwrite(savepath,img)
                #label_show(frame,rectangles)
            else:
                #print("failed ")
                pass
            #cv2.imshow("test",frame)
            key_ = cv2.waitKey(10) & 0xFF
            if key_ == 27 or key_ == ord('q'):
                break
            if frame_cnt == total_num:
                break
    print("total ",idx_cnt)
    v_cap.release()
    cv2.destroyAllWindows()

def detect_faces(img,detector,basename,savedir,cropsize,frame_cnt):
    '''
    img: input image
    basename: save img prefix-name
    basedir: img save dir
    detector: face detector
    '''
    if img is None:
        return 0
    rectangles = detector.detectFace(img)
    #print("detect over",np.shape(rectangles))
    if len(rectangles) == 0:
        return 0
    points = np.array(rectangles)
    points = points[:,5:]
    points_list = points.tolist()
    #crop_imgs = Align_Image.extract_image_chips(frame,points_list)
    crop_imgs = alignImg(img,cropsize,points_list)
    #crop_imgs = alignImg_opencv(frame,crop_size,points_list)
    idx_cnt = 0
    if len(crop_imgs) >0:
        for box_idx,img_out in enumerate(crop_imgs):
            idx_cnt+=1
            f_name = 'f'+str(frame_cnt)
            savepath = os.path.join(savedir,basename+'_'+f_name+'_'+str(box_idx)+".png")
            #cv2.imshow("test",img_out)
            img_out = np.array(img_out,dtype=np.uint8)
            cv2.imwrite(savepath,img_out)
            cv2.waitKey(50) 
        return 1
    else:
        return 0

if __name__ == "__main__":
    #main()
    parm = args()
    p1 = parm.img_path1
    p2 = parm.img_path2
    f_in = parm.file_in
    base_dir = parm.base_dir
    save_dir = parm.save_dir
    base_name = parm.base_name
    cmd_type = parm.cmd_type
    save_dir2 = parm.save_dir2
    img_size = parm.img_size
    size_spl = img_size.strip().split(',')
    img_size = [int(size_spl[0]),int(size_spl[1])]
    #base_name = parm.img_path1
    if cmd_type == 'txtfile':
        save_cropfromtxt(f_in,base_dir,save_dir,img_size,base_name)
    elif cmd_type == 'video':
        save_cropfromvideo(f_in,base_name,save_dir,save_dir2,img_size)
    elif cmd_type == 'imgtest':
        evalu_img(p1,parm.min_size)
    elif cmd_type == 'camera':
        main()
    elif cmd_type == 'showtxt':
        show_formtxt(f_in,parm.min_size)
    else:
        print("No cmd run")
