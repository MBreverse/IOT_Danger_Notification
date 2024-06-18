import sys

# module2_path = '/home/pi/Workspace/tensorflow/tensorflow-2.8.0-cp39-none-linux_aarch64.whl'
# sys.path.append(module2_path)
from tensorflow.keras.models import load_model

import library.DAN as DAN
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.spatial import distance as dist
# from imutils import perspective
# from imutils import contours

# import imutils
import json 
from time import sleep


def read_ref_coordinate():
    # with open('coordinate.json') as f:
    with open('/home/pi/Workspace/user_custom/coordinate.json', 'r') as myfile:
        data=myfile.read()

    coor = json.loads(data)
    x1,y1 = coor["lt"]
    x2,y2 = coor["br"]
    return (x1,y1),(x2,y2)


# get  height and width of the image with nd-array type
def get_h_w(img):

    h = img.shape[0]
    w = img.shape[1]
    return h,w

# compute the mid-point of two point corrdinate
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def danger_degree(D):
    
    thre1 = 100

    thre2 = 200

    if D > thre1:
        return "safe"
    elif D > thre2:
        return "danger"
    else:
        return "very_danger"
import math
def similarity(A,B,h,w):
    errorL2 = cv2.norm( A, B, cv2.NORM_L2 )
    similarity = 1 - errorL2 / ( h * w + 1)
    print('Similarity = ',similarity)
    return similarity

ServerURL = 'http://120.114.183.20:9999'
Reg_addr = None

# IoT名稱
DAN.profile['dm_name'] = 'Raspberry_pi'
DAN.profile['df_list'] = ['A0', ]
DAN.profile['d_name'] = 'RaspberryPi_111'

DAN.device_registration_with_retry(ServerURL, Reg_addr)
print(DAN.get_mac_addr())
sleep(10)

#load model
# model= load_model('/home/pi/Workspace/user_custom/human_model.h5')
model= load_model('/home/pi/Workspace/user_custom/human_model_update.h5')

tl,br = read_ref_coordinate()
print(tl,br)

# construct the HOG feature descriptor
hog = cv2.HOGDescriptor()
print("hog window size:",hog.winSize)

#set HOGSVM detector to find the people
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

count_q = 0
# sys.exit()
cap = cv2.VideoCapture(0)
while(True):
    print()
    sleep(0.4)
        
    # 設置顯示顏色
    colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (0, 125, 125))
       
    ret, cur_frame = cap.read()
    cv2.imwrite('/home/pi/Workspace/pic1.jpg', cur_frame)
    # sleep(0.005)

   
    # copy_frame = cur_frame.copy()
    # cv2.rectangle(copy_frame, tl, br, colors[1], 1)
    
    # cv2.imwrite('/home/pi/Workspace/pic2.jpg', copy_frame)    
    sleep(0.005)

    bg_img = cv2.imread('/home/pi/Workspace/bg.jpg')#png')

    img = cv2.imread('/home/pi/Workspace/pic1.jpg')#png')

    origin_h, origin_w = get_h_w(img)
    print("origin size({},{})".format(origin_h, origin_w))
    
    sim = similarity(img,bg_img,origin_h,origin_w)

    
    total_D =0
    count = 0
    d_avg = 0
    if sim >= 0.92:

        print("detect num of people:{}".format(0))

    else:
        #Calculate the scale of original image r.w.t fixed size
        fixed_size = (origin_h, origin_h)
        # fixed_size = (400, 400)
        scale_h = 1 #origin_h/fixed_size[0]
        scale_w = 1 #origin_w/fixed_size[0]

        bg_frame = cv2.resize(bg_img, fixed_size, interpolation=cv2.INTER_AREA)
        frame = cv2.resize(img, fixed_size, interpolation=cv2.INTER_AREA)

        # print(frame.dtype )

        # avg = (frame + bg_frame )/2

        # frame = frame - avg #+np.ones(frame.shape)
        # frame = frame - bg_frame*0.7
        # frame = frame.astype(np.uint8)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imwrite('/home/pi/Workspace/t.jpg', gray_frame)

        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(gray_frame, winStride=(3,3) )
       
        # convert the format shape of the box
        boxes1 = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        human_boxes = [((x, y), (x + w, y) ,(x + w, y + h),(x, y + h) )for (x, y, w, h) in boxes]
        human_boxes_center = [(x+ w/2., y+h/2.) for (x, y, w, h) in boxes]


        print("detect num of people:{}".format(len(human_boxes)))

        # NN-model post-classify
        remove_list =[]
        iou_size = (400, 400)
        for m, box_m in enumerate(boxes1):
            print(box_m)
            x1,y1,x2,y2 = box_m
            crop_image = img[y1:y2,x1:x2].copy()
            feed_image = crop_image/255.
            feed_image =cv2.resize(feed_image, iou_size, interpolation=cv2.INTER_AREA)
            feed_image = np.expand_dims(feed_image, axis=0)
            pred = model(feed_image)
            class_pred = np.argmax(pred,1)
            print("model post-class",class_pred)
            
            # file_name1 = 'collect_img_{}.jpg'.format(count_q)
          
            # if not  cv2.imwrite(file_name1,crop_image):  
            #     # print(crop_image.shape)         
            #     raise Exception("Could not write image")
          
            # count_q+=1
            # if count_q==20:
            #     print("enough")            
            #     sys.exit()

            #delete the wrong prediction
            if class_pred ==0:
                remove_list.append(m)
               

        
        remove_list.sort(reverse=True)
        for rm_idx in remove_list:
            human_boxes.pop(rm_idx)
            human_boxes_center.pop(rm_idx)

        # set the practical width of the referenced object
        ref_width = 8 # cm

        #referenced object box coordinate setting in frame
        # tl = (85, 247) # top-left
        # br = (113, 452) # bottom-right

        # tl,br = read_ref_coordinate()

        #scaled box
        tl = (tl[0] /scale_w, tl[1] /scale_h)
        br = (br[0] /scale_w, br[1] /scale_h) 

        # compute the width and height of ref_box
        w0 = br[0] - tl[0]
        h0 = br[1] - tl[1]

        #compute the remaining 2 point of ref_box
        tr = (tl[0] + w0 ,tl[1])# top-right
        bl = (tl[0], tl[1] + h0) # bottom-left

        box = np.array((tl,tr,br,bl))

        #compute the center of the ref_box
        cX = tl[0] + w0/2
        cY = tl[1] + h0/2

        D = w0

        # store the coordinate, center, scale of w0 w.r.t. practical width 
        refObj = (box, (cX, cY), D / ref_width)

        color = colors[2]

        #Iterative the box and compute distance and plot the distance from ref_obj to human 
        for i,(center,hbox) in enumerate(zip(human_boxes_center,human_boxes)):
            
            count+=1
            xA,yA = refObj[1]
            xB,yB = center

            #compute the distance between human box and ref_box ,and convert to real distance
            D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
            print((xA,yA,xB,yB))
            (mX, mY) = midpoint((xA, yA), (xB, yB))

            # plot on origin image size
            orig = img.copy()

            origin_xB = xB * scale_w

            notice = danger_degree(D)
            total_D += D
            print("human{} distance {} : {}".format(i,D,notice))
            cv2.putText(orig, "{:.1f} cm".format(D), (int(origin_xB), int(origin_xB - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            origin_box = np.array([(coord[0] * scale_w,coord[1] * scale_h) for coord in hbox])
            origin_ref = np.array([(coord[0] * scale_w,coord[1] * scale_h) for coord in refObj[0]])

            cv2.drawContours(orig, [origin_box.astype("int")], -1, (0, 255, 0), 2)
            cv2.drawContours(orig, [origin_ref.astype("int")], -1, (0, 255, 0), 2)

            # plot line between box
            cv2.circle(orig, (int(xA*scale_w), int(yA*scale_h)), 5, color, -1)
            cv2.circle(orig, (int(xB*scale_w), int(yB*scale_h)), 5, color, -1)
            cv2.line(orig, (int(xA*scale_w), int(yA*scale_h)), (int(xB*scale_w), int(yB*scale_h)), color, 2)

            # disaplay
            cv2.imwrite('/home/pi/Workspace/o.jpg', orig)
            # # cv2.imshow("Image", orig)
            # cv2.waitKey(0)


    if count > 0:
        d_avg = total_D/(count)
        n_total = danger_degree(d_avg)
        print(n_total)


# sys.exit()


    for i in range(1):
        if d_avg!=0:
            print("the avg distance to the ref obj is {} cm".format(d_avg))
        else:
            d_avg =1000
        try:
            DAN.push('A0', int(d_avg))
            print('{} Pushed {}'.format(i+1, d_avg))
        except Exception as e:
            print(e)
            if str(e).find('mac_addr not found:') != -1:
                print('Reg_addr is not found. Try to re-register...')
                DAN.device_registration_with_retry(ServerURL, Reg_addr)
            else:
                print('Connection failed due to unknown reasons.')
                time.sleep(1)


        print()
        sleep(0.5)
