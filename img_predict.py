import os
import pickle
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from math import floor
import io
import requests
from flask_wtf import FlaskForm
from flask import request
from flask import Flask
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField
from wtforms.validators import DataRequired, Length, Email,EqualTo, ValidationError
from flask import render_template, url_for, flash, redirect, request, abort
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import csv
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
def getTamilChar(tamilCharacterCode, indx):
       return tamilCharacterCode[indx]
tamilCharacterCode = []
with open('/home/snekha/datasets/tamil_data/unicodeTamil.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    for i in data:
        go = i[1].split(' ')
        charL = ""
        for gg in go:
            charL = charL + "\\u"+str(gg)
        tamilCharacterCode.append(charL.encode('utf-8').decode('unicode-escape'))
import cv2
from imutils import contours

# Load image, grayscale, Otsu's threshold
imag = cv2.imread('final.png')

# apply image thresholding
# thresh = cv2.adaptiveThreshold(imag,    
#           255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# invert the image, 255 is the maximum value
# thresh = 255 — thresh
gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
# imag = cv2.adaptiveThreshold(imag, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, -15)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
# imag = align_text(imag)
# rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
# threshed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)
# ret, thresh = cv2.threshold(imag, 100, 255, cv2.THRESH_TOZERO)
# thresh = cv2.resize(thresh, (960, 540)) 
# Find contours, sort from left-to-right, then crop
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts, _ = contours.sort_contours(cnts, method="left-to-right")


ROI_number = 0

for c in cnts:
    area = cv2.contourArea(c)
    if area > 10:
        x,y,w,h = cv2.boundingRect(c)
        ROI = 255 - imag[y:y+h, x:x+w]
        cv2.imwrite('/home/snekha/hackathons/taml/ml/code/image/ROI_{}.png'.format(ROI_number), ROI)
    
                # align image text
                  
        # image_path = '/home/snekha/hackathons/taml/ml/code/image/ROI_{}.png'.format(ROI_number)
        # image_file = Image.open(image_path)
        # image_file.save('/home/snekha/hackathons/taml/ml/code/image/ROI_{}.png'.format(ROI_number), quality=95)
        cv2.rectangle(imag, (x, y), (x + w, y + h), (36,255,12), 2) 
        model = load_model('/home/snekha/datasets/tamil_data/tamilALLEzhuthukalKeras_Model.h5')
        model1 = load_model('/home/snekha/datasets/tamil_data/upscaling.h5')
        # model = None
        
        datasetsLoc = '/home/snekha/hackathons/taml/ml/code/image/'
        import PIL 
        w,h = 128,128
        i = 0
        shapeL=[]
        import cv2
        # from imutils import contours
        images=[]
        labels=[]
        # from keras.utils.np_utils import to_categorical  
        # encoder = OneHotEncoder(categories='auto')
        def bbox2(img1):
            img = 1 - img1
            rows = np.any(img, axis=1)
            cols = np.any(img, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            return rmin, rmax, cmin, cmax
        def RR(img):
            rmin, rmax, cmin, cmax = bbox2(img)
            # print(rmin, rmax, cmin, cmax)
            npArr = img[rmin:rmax, cmin:cmax]
            npArr = cv2.resize(npArr, dsize=(100, 100))
            jinga = np.ones((128,128))
            jinga[14:114,14:114] = npArr
            npArr = jinga.reshape(128, 128 , 1)
            return npArr

        def getTamilChar(tamilCharacterCode, indx):
            return tamilCharacterCode[indx]

        def plotIm(img_):
          plt.imshow(img_, cmap='gray')
          plt.show()

        def init_somethings():

            global tamilCharacterCode, model

            with open('/home/snekha/datasets/tamil_data/unicodeTamil.csv', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
                for i in data:
                    go = i[1].split(' ')
                    charL = ""
                    for gg in go:
                        charL = charL + "\\u"+str(gg)
                        tamilCharacterCode.append(charL.encode('utf-8').decode('unicode-escape'))

        # for folders in os.listdir(datasetsLoc):
            # for files in  os.listdir(datasetsLoc):   
        def load_image(path, downfactor=3):
            original_img = load_img(path)
            downscaled = original_img.resize((original_img.size[0] // downfactor,
                original_img.size[1] // downfactor), Image.BICUBIC)
            return (original_img, downscaled)

        def get_y_channel(image):
            ycbcr = image.convert("YCbCr")
            (y, cb, cr) = ycbcr.split()
            y = np.array(y)
            y = normalize(y.astype("float32"))
            return (y, cb, cr)

        def upscale_image(img, model):
            y, cb, cr = get_y_channel(img)
            input = np.expand_dims(y, axis=0)
            out = model.predict(input)[0]
            out *= 255.0
            out = out.clip(0, 255)
            out = out.reshape((np.shape(out)[0], np.shape(out)[1]))
            new_y = Image.fromarray(np.uint8(out), mode="L") # it will generate y channel of image
            new_cb = cb.resize(new_y.size, PIL.Image.BICUBIC) 
            new_cr = cr.resize(new_y.size, PIL.Image.BICUBIC)
            res = PIL.Image.merge("YCbCr", (new_y, new_cb, new_cr)).convert(
                "RGB"
            ) # it will convert from YCbCr to RGB image
            return res
        # test_path="/home/snekha/hackathons/taml/ml/code/image//ROI_{}.png".format(ROI_number)
        # # for file_name in os.listdir("/home/snekha/hackathons/taml/ml/code/image/"):
        # # test_path = os.path.join(test_dir, file_name)
        # original_img, downscaled = load_image(test_path)
        # res = upscale_image(downscaled, model)
        # cv2.imwrite(test_path, res)
        # the default
        image_path = '/home/snekha/hackathons/taml/ml/code/image/ROI_{}.png'.format(ROI_number)
        image_file = Image.open(image_path)
        image_file.save('/home/snekha/hackathons/taml/ml/code/image/ROI_{}.png'.format(ROI_number), quality=95)
        image = Image.open(datasetsLoc+'/'+'ROI_{}.png'.format(ROI_number))
        thresh = 200
        fn = lambda x : 255 if x > thresh else 0
        image=image.convert('L')
        img=np.asarray(image, dtype=np.uint8)
        shapeL.append(img.shape)
        img2 = RR(img)
        img2 = np.asarray(img2, dtype=np.uint8)
                    # img2 = encoder.fit_transform(img2)
        images.append(img2)

        # filIm = open('/home/snekha/hackathons/taml/ml/code/new_img/image_ALL_128x128.obj', 'wb')
        # pickle.dump(images, filIm)
        # numCategory = 156
        # filIm = open('/home/snekha/hackathons/taml/ml/code/new_img/image_ALL_128x128.obj', 'rb')
        # images = pickle.load(filIm)
        npArr=np.array(images)

        # print(images)
        # npArr=images[0]
        # npArr = np.asarray(images, dtype=np.uint8)
        npArr = npArr.reshape(1, 128, 128 , 1)
        # print(npArr)
        # # npArr=images
        atc = model.predict(npArr)

        percentage = atc[0]
        valsss = atc[0].argsort()[-3:][::-1]
        responseTextSt = getTamilChar(tamilCharacterCode,valsss[0])+","+ getTamilChar(tamilCharacterCode,valsss[1])+ ","+ getTamilChar(tamilCharacterCode,valsss[2])
        responseTextSt = responseTextSt + ',%.3f,%.3f,%.3f'%(percentage[valsss[0]] *100.0,percentage[valsss[1]] *100.0,percentage[valsss[2]]*100.0)
        # print(responseTextSt)
        a=getTamilChar(tamilCharacterCode,valsss[0])
        print(a)
        file1 = open("tamil_pdf.txt", "a")  # append mode
        file1.write(a)
        file1.close()
        ROI_number += 1