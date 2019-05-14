# auther : lxy
# time : 2017.12.28 /09:56
#project:
# tool: python2
#version: 0.1
#modify:
#name:
#citations:
#############################
import numpy as np
import os
os.environ['KERAS_BACKEND']='tensorflow'
import argparse
import cv2
import sys
sys.path.append('../../')
#####################
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model
import time
#############

use_gender=0

def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default='../../trained_models/ag_models/weights.25000-0.03.hdf5',
                        help="path to weight file (e.g. weights.18-4.06.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=2,
                        help="width of network")
    parser.add_argument("--img_size",type=int,default=64,
                        help='the net input size')
    parser.add_argument('--json_fil',type=str,default='../../trained_models/ag_models/WRN_16_2.json',
                        help='saved json file')
    parser.add_argument('--file_in',type=str,default='None',
                        help='input file')
    args = parser.parse_args()
    return args


def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}
    elif dataset_name == 'imdb':
        return {0:'woman', 1:'man'}
    elif dataset_name == 'KDEF':
        return {0:'AN', 1:'DI', 2:'AF', 3:'HA', 4:'SA', 5:'SU', 6:'NE'}
    else:
        raise Exception('Invalid dataset name')

def draw_label(image, point, label,color=(255,255,255),mode=0, font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
               font_scale=1, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = int(point[0]),int(point[1])
    w, h = int(point[2]),int(point[3])
    lb_w = int(size[0])
    lb_h = int(size[1])
    if mode ==1:
        if y-int(2*lb_h) <= 0:
            cv2.rectangle(image, (x, y +h+lb_h), (x + lb_w, y+h+2*lb_h), color)
            #point:left_down
            cv2.putText(image, label, (x,y+h+2*lb_h), font, font_scale, color, thickness) 
        else:
            cv2.rectangle(image, (x, y-2*lb_h), (x + lb_w, y-lb_h), color)
            cv2.putText(image, label, (x,y-lb_h), font, font_scale, color, thickness)
    elif mode == 2:
        if y-int(3*lb_h) <= 0:
            cv2.rectangle(image, (x, y+h+2*lb_h), (x + lb_w, y+h+3*lb_h), color)
            cv2.putText(image, label, (x,y+h+3*lb_h), font, font_scale, color, thickness)
        else:
            cv2.rectangle(image, (x, y-3*lb_h), (x + lb_w, y-2*lb_h), color)
            cv2.putText(image, label, (x,y-2*lb_h), font, font_scale, color, thickness) 
    else:
        if y-lb_h <= 0:
            cv2.rectangle(image, (x,y +h), (x + lb_w, y+h+lb_h), color)
            cv2.putText(image, label, (x,y+h+lb_h), font, font_scale, color, thickness)
        else:
            cv2.rectangle(image, (x, y-lb_h), (x + lb_w, y), color)
            cv2.putText(image, label, (x,y), font, font_scale, color, thickness) 
    #cv2.FONT_HERSHEY_SIMPLEX
    '''
    if y-int(size[1]) <= 0:
        cv2.rectangle(image, (x, y +h), (x + int(size[0]), y+h+int(size[1])), (255, 0, 0))
        cv2.putText(image, label, (x,y+h+size[1]), font, font_scale, color, thickness)
    else:
        cv2.rectangle(image, (x, y-int(size[1])), (x + int(size[0]), y), (255, 0, 0))
        cv2.putText(image, label, (x,y), font, font_scale, color, thickness)
    '''

def draw_box(img,box,color=(255,0,0)):
    (row,col,cl) = np.shape(img)
    #b = board_img(box,col,row)
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]+box[0]), int(box[3]+box[1])), color)
'''  
def board_img(boxes,wid,height):
    #print ('box shape ',np.shape(boxes))
    #print boxes
    [x1,y1,x2,y2] = boxes[:4]
    x1 -= 20
    y1 -= 20
    x2 += 20
    y2 += 20
    x1 = np.int(np.max(x1,0))
    y1 = np.int(np.max(y1,0))
    x2 = np.int(min(x2,wid))
    y2 = np.int(min(y2,height))
    return [x1,y1,x2-x1,y2-y1]
'''
def board_img(boxes,wid,height):
    #print ('box shape ',np.shape(boxes))
    #print boxes
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3],
    offset_w = (x2-x1)/5.0
    offset_h = (y2-y1)/5.0
    x1 -= offset_w
    y1 -= 3*offset_h
    x2 += offset_w
    y2 += offset_h
    x1 = map(int,np.maximum(x1,0))
    y1 = map(int,np.maximum(y1,0))
    x2 = map(int,np.minimum(x2,wid-1))
    y2 = map(int,np.minimum(y2,height-1))
    #box = [x1,y1,x2,y2,boxes[:,4]]
    box = [x1,y1,x2,y2]
    box = np.vstack(box)
    return box.T

def detect_face(img,img_size,offsets,detection_model):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_image = img
    faces = detection_model.detectMultiScale(gray_image, 1.3, 5,minSize=(50,50),maxSize=(200,200))
    #results = detector.detect_face(img)
    (row,col,cl) = np.shape(img)
    if len(faces)==0:
        return np.array([]),np.array([])
    total_boxes = []
    age_facecrops = []
    gender_facecrops = []
    emotion_facecrops = [] 
    age_offsets = offsets[0]
    gender_offsets = offsets[1]
    emotion_offsets = offsets[2]
    age_size,gender_size,emotion_size = img_size
    for face_coordinates in faces :
        x1, x2, y1, y2 = apply_offsets(face_coordinates, age_offsets,col,row)
        '''
        if x2 <=x1 or y2<=y1 :
            continue
        '''
        gender_x1, gender_x2, gender_y1, gender_y2 = apply_offsets(face_coordinates, gender_offsets,col,row)
        emt_x1, emt_x2, emt_y1, emt_y2 = apply_offsets(face_coordinates, emotion_offsets,col,row)
        try:
            gender_face = rgb_image[gender_y1:gender_y2, gender_x1:gender_x2]
            emt_face = gray_image[emt_y1:emt_y2, emt_x1:emt_x2]
            age_face = gray_image[y1:y2, x1:x2]
            gender_face = cv2.resize(gender_face,(gender_size,gender_size))
            emt_face = cv2.resize(emt_face, (emotion_size,emotion_size))
            age_face = cv2.resize(age_face, (age_size,age_size))
        except:
            continue
        gender_face = (gender_face -127.5)*0.0078125
        #gender_face = np.expand_dims(gender_face,0)
        gender_facecrops.append(gender_face)
        emt_face = (emt_face -127.5)*0.0078125
        #emt_face = np.expand_dims(emt_face,0)
        emt_face = np.expand_dims(emt_face,-1)
        emotion_facecrops.append(emt_face)
        age_face = (age_face -127.5)*0.0078125
        #age_face = np.expand_dims(age_face,0)
        age_face = np.expand_dims(age_face,-1)
        age_facecrops.append(age_face)
        total_boxes.append([x1,y1,x2-x1,y2-y1])
    #total_boxes = results[0]
    #points = results[1]
    #print len(total_boxes)
    '''
    draw = img.copy()
    if len(total_boxes) !=0:
       
        for i in range(len(total_boxes)):
            b = board_img(total_boxes[i],col,row)
        
        for b in total_boxes:
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]+b[0]), int(b[3]+b[1])), (255, 255, 255))
        
        for p in points:
            for i in range(5):
                cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), 2)
    else:
        return np.array([]),np.array([]),np.array([])
    '''
    if len(total_boxes) !=0:
        total_boxes = np.array(total_boxes).astype(np.int)
        gender_facecrops = np.array(gender_facecrops).astype(np.float32)
        emotion_facecrops = np.array(emotion_facecrops).astype(np.float32)
        age_facecrops = np.array(age_facecrops).astype(np.float32)
        face_crops = [age_facecrops,gender_facecrops,emotion_facecrops]
        #print("right")
        return face_crops, total_boxes
    else:
        #print("none")
        return np.array([]),np.array([])

def apply_offsets(face_coordinates, offsets,w,h):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    #print(x+width+x_off,w)
    x1 = np.maximum(x-x_off,0)
    x_n = x+width+x_off
    x2 = np.minimum(x_n,w)
    y1 = np.maximum(y-y_off,0)
    y_n = y+height+y_off
    y2 = np.minimum(y_n,h)
    return (x1,x2,y1,y2)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def main():
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file
    age_size = args.img_size
    json_file = args.json_fil
    file_in = args.file_in
    detect_model_path = "../../trained_models/detection_models/haarcascade_frontalface_default.xml"
    emotion_model_path = '../../trained_models/emotion_models/fer2013_mini_XCEPTION.110-0.65.hdf5'
    #gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
    #gender_model_path = '../trained_models/gender_models/gender_mini_XCEPTION.176-0.91.hdf5'
    gender_model_path = '../../trained_models/gender_models/gender_mini_XCEPTION.2878-0.93.hdf5'
    gender_offsets = (30, 60)
    emotion_offsets = (20, 40)
    age_offsets =(0,0)
    offsets_list = [age_offsets,gender_offsets,emotion_offsets]
    #get labels for gender and emotion
    emotion_labels = get_labels('fer2013')
    gender_labels = get_labels('imdb')
    # load model and weights
    #model = WideResNet(img_size, depth=depth, k=2)()
    emotion_classifier = load_model(emotion_model_path, compile=False)
    gender_classifier = load_model(gender_model_path, compile=False)
    json_f = open(json_file,'r')
    json_c = json_f.read()
    model = model_from_json(json_c)
    model.summary()
    model.load_weights(weight_file)
    #get gender and emotion size
    emotion_target_size = emotion_classifier.input_shape[1:3]
    gender_target_size = gender_classifier.input_shape[1:3]
    img_size= (age_size,gender_target_size[0],emotion_target_size[0])
    #detector = MtcnnDetector(model_folder='./model', ctx=mx.cpu(0), num_worker = 1 , accurate_landmark = False)
    detector = load_detection_model(detect_model_path)
    # capture video
    if file_in =='None':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(file_in)
        print("using file")
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow("result")

    label_count = 0
    age_sum = 0
    first_face_count = 0
    #gender_l = np.zeros([2])
    print("cap.isOpened()",cap.isOpened())
    while cap.isOpened():
        # get video frame
        ret, img = cap.read()
        if not ret:
            print("error: failed to capture image")
            return -1
        if img is not None:
            img = cv2.resize(img, (640,480))
            draw = img
        t1 = time.time()
        face_crops,bboxes = detect_face(img,img_size,offsets_list,detector)
        t2 = time.time()
        print("face detect ",t2-t1)
        #print(np.shape(bboxes))
        #input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img_h, img_w, _ = np.shape(input_img)
        if len(face_crops) > 0:
            # predict ages and genders of the detected faces
            #print("face shape ",np.shape(faces))
            '''
            face_gray = []
            for i in range(len(faces)):
                face = np.expand_dims(cv2.cvtColor(faces[i,:,:,:],cv2.COLOR_BGR2GRAY),-1)
                face_gray.append(face)
            face_gray = np.array(face_gray)
            '''
            age_faces = face_crops[0]
            gender_faces = face_crops[1]
            emotion_faces = face_crops[2]
            t1 = time.time()
            age_pred = model.predict(age_faces)
            gender_pred = gender_classifier.predict(gender_faces)
            emotion_pred = emotion_classifier.predict(emotion_faces)
            t2 = time.time()
            print("pred time ",t2-t1)
            gender_label_pred = np.argmax(gender_pred,axis=1)
            emotion_label_pred = np.argmax(emotion_pred,axis=1)
            if use_gender==1 :
                predicted_genders = gender_pred
            else:
                #predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = age_pred.dot(ages).flatten()
                predicted_ages = map(int,predicted_ages)
            first_face_count +=1
            if first_face_count == 5:
                first_face_count =2
        else:
            continue
        # draw results
        if ret :
            label_count+=1
            if label_count == 60:
                num = bboxes.shape[0]
                gender_cnt = np.zeros([num,2])
                emt_cnt = np.zeros([num,7])
                age_cnt = np.zeros([num,101])
                label_count =0
            if first_face_count ==1:
                num = bboxes.shape[0]
                gender_cnt = np.zeros([num,2])
                emt_cnt = np.zeros([num,7])
                age_cnt = np.zeros([num,101])
        #ages_pre = predicted_ages

        for i,box in enumerate(bboxes[:num]):
            if use_gender==1 :
                label = "{}".format("F" if predicted_genders[i][0] > 0.5 else "M")
            else:
                #label = "{}, {}".format(int(predicted_ages[i]),\
                                    #"F" if predicted_genders[i][0] > 0.5 else "M")
                #gender_text = gender_labels[gender_label_pred[i]]
                #emotion_text = emotion_labels[emotion_label_pred[i]]
                #if gender_text == gender_labels[0]:
                 #   color = (0, 0, 255)
                #else:
                  #  color = (255, 0, 0)
                #do hist for all results
                #print("age ",age_cnt.shape,predicted_ages[i])
                gender_cnt[i,gender_label_pred[i]] +=1
                emt_cnt[i,emotion_label_pred[i]] +=1
                age_cnt[i,predicted_ages[i]]+=1
                #draw face boxs
                #draw_box(draw,box,color)
                #draw_label(draw,bboxes[i],gender_text,color,1)
                #draw_label(draw,bboxes[i],emotion_text,color,0)
                if label_count == 50:
                    label = "{}".format(int(predicted_ages[i]))
                    gder_max = np.argmax(gender_cnt[i,:])
                    gender_text = gender_labels[gder_max]
                    emt_max = np.argmax(emt_cnt[i,:])
                    emotion_text = emotion_labels[emt_max]
                    if gender_text == gender_labels[0]:
                        color = (0, 0, 255)
                    else:
                        color = (255, 0, 0)
                if first_face_count == 1:
                    label = "{}".format(int(predicted_ages[i]))
                    gender_text = gender_labels[gender_label_pred[i]]
                    emotion_text = emotion_labels[emotion_label_pred[i]]
                    if gender_text == gender_labels[0]:
                        color = (0, 0, 255)
                    else:
                        color = (255, 0, 0)
                    draw_label(draw, bboxes[i], label,color,2)
                    draw_box(draw,box,color)
                    draw_label(draw,bboxes[i],gender_text,color,1)
                    draw_label(draw,bboxes[i],emotion_text,color,0)
                else:
                    draw_label(draw, bboxes[i], label,color,2)
                    draw_box(draw,box,color)
                    draw_label(draw,bboxes[i],gender_text,color,1)
                    draw_label(draw,bboxes[i],emotion_text,color,0)
            #draw_label(img, (bboxes[i][0], bboxes[i][1]), label)
        cv2.imshow("result", draw)
        #key = cv2.waitKey(30)
        if (cv2.waitKey(1)& 0xFF) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    set_keras_backend('tensorflow')
    main()
    cv2.destroyAllWindows()
