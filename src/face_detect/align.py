# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/07/5 10:09
#project: Face detect
#company: 
#rversion: 0.1
#tool:   python 3
#modified:
#description  crop face align
####################################################
import numpy as np
import cv2
import math
from skimage import transform as trans

class Align_img(object):
    def __init__(self,desired_size,padding=0):
        self.h,self.w = desired_size
        self.padding = padding

    def list2colmatrix(self, pts_list):
        """
            convert list to column matrix
        Parameters:
        ----------
            pts_list:
                input list
        Retures:
        -------
            colMat:
        """
        assert len(pts_list) > 0
        colMat = []
        for i in range(len(pts_list)):
            colMat.append(pts_list[i][0])
            colMat.append(pts_list[i][1])
        #print("the colMat shape before mat ",np.shape(colMat))
        colMat = np.matrix(colMat).transpose()
        #print("the colMat shape after mat ",np.shape(colMat))
        return colMat

    def find_tfrom_between_shapes(self, from_shape, to_shape):
        """
            find transform between shapes
        Parameters:
        ----------
            from_shape:
            to_shape:
        Retures:
        -------
            tran_m:
            tran_b:
        """
        assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0

        sigma_from = 0.0
        sigma_to = 0.0
        cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])

        # compute the mean and cov
        from_shape_points = from_shape.reshape(from_shape.shape[0]/2, 2)
        to_shape_points = to_shape.reshape(to_shape.shape[0]/2, 2)
        mean_from = from_shape_points.mean(axis=0)
        mean_to = to_shape_points.mean(axis=0)

        for i in range(from_shape_points.shape[0]):
            temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
            sigma_from += temp_dis * temp_dis
            temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
            sigma_to += temp_dis * temp_dis
            cov += (to_shape_points[i].transpose() - mean_to.transpose()) * (from_shape_points[i] - mean_from)

        sigma_from = sigma_from / to_shape_points.shape[0]
        sigma_to = sigma_to / to_shape_points.shape[0]
        cov = cov / to_shape_points.shape[0]

        # compute the affine matrix
        s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
        u, d, vt = np.linalg.svd(cov)

        if np.linalg.det(cov) < 0:
            if d[1] < d[0]:
                s[1, 1] = -1
            else:
                s[0, 0] = -1
        r = u * s * vt
        c = 1.0
        if sigma_from != 0:
            c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

        tran_b = mean_to.transpose() - c * r * mean_from.transpose()
        tran_m = c * r

        return tran_m, tran_b

    def extract_image_chips(self, img, points):
        """
            crop and align face
        Parameters:
        ----------
            img: numpy array, bgr order of shape (1, 3, n, m)
                input image
            points: numpy array, n x 10 (x1,y1, x2,y2... x5,y5)
            desired_size: default 256
            padding: default 0
        Retures:
        -------
            crop_imgs: list, n
                cropped and aligned faces
        """
        crop_imgs = []
        for p in points:
            shape = [] #p
            for k in range(len(p)/2):
                shape.append(p[k])
                shape.append(p[k+5])
            if self.padding > 0:
                padding = self.padding
            else:
                padding = 0
            # average positions of face points
            mean_face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
            mean_face_shape_y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]

            from_points = []
            to_points = []

            for i in range(len(shape)/2):
                x = (padding + mean_face_shape_x[i]) / (2 * padding + 1) * self.w
                y = (padding + mean_face_shape_y[i]) / (2 * padding + 1) * self.h
                to_points.append([x, y])
                from_points.append([shape[2*i], shape[2*i+1]])

            # convert the points to Mat
            #print("the points from and to shape ",np.shape(from_points),np.shape(to_points))
            from_mat = self.list2colmatrix(from_points)
            to_mat = self.list2colmatrix(to_points)
            #print("the points mat from and to shape ",np.shape(from_mat),np.shape(to_mat))
            # compute the similar transfrom
            tran_m, tran_b = self.find_tfrom_between_shapes(from_mat, to_mat)

            probe_vec = np.matrix([1.0, 0.0]).transpose()
            probe_vec = tran_m * probe_vec

            scale = np.linalg.norm(probe_vec)
            angle = 180.0 / math.pi * math.atan2(probe_vec[1, 0], probe_vec[0, 0])
            #print("the angle is ",angle)
            from_center = [(shape[0]+shape[2])/2.0, (shape[1]+shape[3])/2.0]
            to_center = [0, 0]
            to_center[1] = self.h * 0.4
            to_center[0] = self.w * 0.5

            ex = to_center[0] - from_center[0]
            ey = to_center[1] - from_center[1]

            rot_mat = cv2.getRotationMatrix2D((from_center[0], from_center[1]), -1*angle, scale)
            rot_mat[0][2] += ex
            rot_mat[1][2] += ey

            chips = cv2.warpAffine(img, rot_mat, (self.w, self.h))
            if chips is not None:
                crop_imgs.append(chips)
        return crop_imgs

def alignImg(img,image_size,points):
    '''
    image_size: [h,w]
    points: coordinates of the 5 points(eye,nose,mouse) will be ordered x1,x2,x3,x4,x5,y1,y2,y3,y4,y5
    '''
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
        src[:,0] += 8.0
    tform = trans.SimilarityTransform()
    crop_imgs = []
    cropsize = (image_size[1],image_size[0])
    for p in points:
        dst = np.reshape(p,(2,5)).T
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderMode=1)
        #tform = trans.estimate_transform('affine', dst, src) # Assume square
        #warped = cv2.warpPerspective(img, tform.params, cropsize, borderMode=1)
        if warped is not None:
            crop_imgs.append(warped)
    return crop_imgs


def alignImg_opencv(img,image_size,points):
    '''
    image_size: [h,w]
    points: coordinates of the 5 points(eye,nose,mouse) will be ordered x1,x2,x3,x4,x5,y1,y2,y3,y4,y5
    '''
    dst = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
        dst[:,0] += 8.0
    crop_imgs = []
    #new_dst_points = np.array([[(dst[0,0] + dst[1,0]) / 2, (dst[0,1] + dst[1,1]) / 2], [(dst[3,0] + dst[4,0]) / 2, (dst[3,1] + dst[4,1]) / 2]])
    for p in points:
        src = np.reshape(p,(2,5)).T
        src = src.astype(np.float32)
        #print(src.shape,dst.shape)
        #src_points = np.array([[(src[0][0] + src[1][0]) / 2, (src[0][1] + src[1][1]) / 2],[(src[3][0] + src[4][0]) / 2, (src[3][1] + src[4][1]) / 2]])
        similarTransformation = cv2.estimateRigidTransform(src.reshape(1,5,2), dst.reshape(1,5,2), fullAffine=True)
        #similarTransformation = cv2.getPerspectiveTransform(src,dst)
        #if similarTransformation is None:
            #continue
        warped = cv2.warpAffine(img, similarTransformation,(image_size[1],image_size[0]),borderMode=1)
        if warped is not None:
            crop_imgs.append(warped)
    return crop_imgs

def alignImg_angle(input_image,output_size,point_list,ec_mc_y=40):
    '''
    output_size: [h,w]
    points: coordinates of the 5 points(eye,nose,mouse) will be ordered x1,x2,x3,x4,x5,y1,y2,y3,y4,y5
    '''
    crop_imgs = []
    for points in point_list:
        points = np.reshape(points,(2,5)).T
        points = points.astype(np.float32)
        eye_center = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
        mouth_center = ((points[3][0] + points[4][0]) / 2, (points[3][1] + points[4][1]) / 2)
        angle = math.atan2(mouth_center[0] - eye_center[0], mouth_center[1] - eye_center[1]) / math.pi * -180.0
        # angle = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]) / math.pi * 180.0
        scale = ec_mc_y / math.sqrt((mouth_center[0] - eye_center[0])**2 + (mouth_center[1] - eye_center[1])**2)
        center = ((points[0][0] + points[1][0] + points[3][0] + points[4][0]) / 4, (points[0][1] + points[1][1] + points[3][1] + points[4][1]) / 4)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        rot_mat[0][2] -= (center[0] - output_size[1] / 2)
        rot_mat[1][2] -= (center[1] - output_size[0] / 2)
        warp_dst = cv2.warpAffine(input_image,rot_mat,(output_size[1],output_size[0]))
        if warp_dst is not None:
            crop_imgs.append(warp_dst)
    return crop_imgs

def solve_transform_org(src,dst,n=5):
    '''
    src: [[x1,y1],[x2,y2],..[x5,y5]]
    dst: [[x1,y1],[x2,y2],..[x5,y5]]
    '''
    a = np.zeros([12*n])
    b = np.zeros([2*n])
    for i in range(n):
        j = i*12
        k = i*12+6
        a[j] = a[k+3] = src[i][0]
        a[j+1] = a[k+4] = src[i][1]
        a[j+2] = a[k+5] = 1
        a[k] = a[k+1] = a[k+2] =0
        b[i*2] = dst[i][0]
        b[i*2+1] = dst[i][1]
    row = int(2*n)
    a_s = np.reshape(a,(row,6))
    #a_append = np.zeros((row,4))
    #a_s = np.hstack([a_s,a_append])
    b_s = np.reshape(b,(row,1))
    #print(a_s,b_s)
    _,M = cv2.solve(a_s,b_s,flags=cv2.DECOMP_SVD)
    return M.reshape([2,3])

def solve_transform(refpoints, points, w = None):
    if w == None:
        w = [1] * (len(points) * 2)
    assert(len(w) == 2*len(points))
    y = []
    for n, p in enumerate(refpoints):
        y += [p[0]/w[n*2], p[1]/w[n*2+1]]
    A = []
    for n, p in enumerate(points):
        A.extend([ [p[0]/w[n*2], p[1]/w[n*2], 0, 0, 1/w[n*2], 0], [0, 0, p[0]/w[n*2+1], p[1]/w[n*2+1], 0, 1/w[n*2+1]] ])
    #print(np.shape(A))
    lstsq = cv2.solve(np.array(A), np.array(y), flags=cv2.DECOMP_SVD)
    h11, h12, h21, h22, dx, dy = lstsq[1]
    #err = 0#lstsq[1]
    #R = np.array([[h11, h12, dx], [h21, h22, dy]])
    # The row above works too - but creates a redundant dimension
    R = np.array([[h11[0], h12[0], dx[0]], [h21[0], h22[0], dy[0]]])
    return R

def alignImg_solve(img,output_size,points):
    '''
    output_size: [h,w]
    points: coordinates of the 5 points(eye,nose,mouse) will be ordered x1,x2,x3,x4,x5,y1,y2,y3,y4,y5
    '''
    dst = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if output_size[1]==112:
        dst[:,0] += 8.0
    crop_imgs = []
    i=0
    #new_dst_points = np.array([[(dst[0,0] + dst[1,0]) / 2, (dst[0,1] + dst[1,1]) / 2], [(dst[3,0] + dst[4,0]) / 2, (dst[3,1] + dst[4,1]) / 2]])
    for p in points:
        src = np.reshape(p,(2,5)).T
        src = src.astype(np.float32)
        #src_points = np.array([[(src[0][0] + src[1][0]) / 2, (src[0][1] + src[1][1]) / 2],[(src[3][0] + src[4][0]) / 2, (src[3][1] + src[4][1]) / 2]])
        M = solve_transform_org(src, dst)
        #if similarTransformation is None:
            #continue
        #print(M)
        warped = cv2.warpAffine(img, M,(output_size[1],output_size[0]),borderMode=1)
        if warped is not None:
            crop_imgs.append(warped)
    return crop_imgs