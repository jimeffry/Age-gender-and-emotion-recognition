import numpy as np 
from easydict import EasyDict

config = EasyDict()
# if rnet_out=1,face detect result will be output by Rnet
config.rnet_out = 0
# if onet_out=1,face detect result will be output by Onet
config.onet_out = 0
# if pnet_out=1,face detect result will be output by Pnet
config.pnet_out = 0
#if time=1, print the time consuming by every net
config.time = 0
#if crop_org=1, the main file--test.py will directly output the result by giving face detect model
config.crop_org = 0
#if x_y=1, coordinates of the 5 points(eye,nose,mouse) will be ordered x1,y1,x2,y2,...,x5,y5
#if x_y=0, coordinates of the 5 points(eye,nose,mouse) will be ordered x1,x2,x3,x4,x5,y1,y2,y3,y4,y5
config.x_y = 0
#if box_widen = 1, the boxes got by face detection model output , will be widened. used for images to build database
config.box_widen = 0
config.id_box_widen = 0
#whether to downsample img to short size is 320
config.img_downsample = 0
#whether to keep the original ratio to get ouput size
config.imgpad = 0
# whether to show the picture
config.show = 1