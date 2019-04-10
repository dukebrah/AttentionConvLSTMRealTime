# author:    Adrian Rosebrock
# website:   http://www.pyimagesearch.com

# USAGE
# BE SURE TO INSTALL 'imutils' PRIOR TO EXECUTING THIS COMMAND
# python fps_demo.py
# python fps_demo.py --display 1

# import the necessary packages
from __future__ import print_function
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
import pandas as pd

def putIterationsPerSec(frame, label):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "class label: "+label,
        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

# created a *threaded *video stream, allow the camera senor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = VideoStream(src=0).start()
fps = FPS().start()


# In[2]:

import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import io
import sys
sys.path.append("./networks")
import numpy as np
import tensorflow as tf
keras=tf.contrib.keras
l2=keras.regularizers.l2
K=tf.contrib.keras.backend
import inputs as data
from res3d_clstm_mobilenet import res3d_clstm_mobilenet
from datagen import jesterTestImageGenerator
from datagen import isoTestImageGenerator
from datetime import datetime
import cv2
import numpy as np
import sys

# Modality
RGB = 0
Depth = 1
Flow = 2
# Dataset
JESTER = 0
ISOGD = 1

cfg_modality = RGB
cfg_dataset = JESTER

if cfg_modality==RGB:
  str_modality = 'rgb'
elif cfg_modality==Depth:
  str_modality = 'depth'
elif cfg_modality==Flow:
  str_modality = 'flow'

if cfg_dataset==JESTER:
  seq_len = 16
  batch_size = 16
  num_classes = 27
  testing_datalist = './dataset_splits/Jester/valid_%s_list.txt'%str_modality
elif cfg_dataset==ISOGD:
  seq_len = 32
  batch_size = 8
  num_classes = 249
  testing_datalist = './dataset_splits/IsoGD/valid_%s_list.txt'%str_modality

weight_decay = 0.00005
model_prefix = '/home/dineshp/AttentionConvLSTM/models/'
  
inputs = keras.layers.Input(shape=(seq_len, 112, 112, 3))
feature = res3d_clstm_mobilenet(inputs, seq_len, weight_decay)
flatten = keras.layers.Flatten(name='Flatten')(feature)
classes = keras.layers.Dense(num_classes, activation='linear', kernel_initializer='he_normal',
	            kernel_regularizer=l2(weight_decay), name='Classes')(flatten)
outputs = keras.layers.Activation('softmax', name='Output')(classes)
model = keras.models.Model(inputs=inputs, outputs=outputs)
optimizer = keras.optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

if cfg_dataset==JESTER:
  pretrained_model = '%s/jester_%s_gatedclstm_weights.h5'%(model_prefix,str_modality)
elif cfg_dataset==ISOGD:
  pretrained_model = '%s/isogr_%s_gatedclstm_weights.h5'%(model_prefix,str_modality)
print('Loading pretrained model from %s' % pretrained_model)
model.load_weights(pretrained_model, by_name=False)

labels = pd.read_csv('jester-v1-labels.csv', header=None) 

pred = "";
bufferf = []

# loop over some frames...this time using the threaded stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = putIterationsPerSec(frame, pred)
	
	#frame = cv2.resize(frame, (640, 480)) 

	crop_img = frame[0:720, 0:720]
	#h,w = crop_img.shape[:2]
	input_img = cv2.resize(crop_img,(112,112))
	cv2.imshow("Frame", crop_img)
	key = cv2.waitKey(1) & 0xFF
	if(np.shape(np.asarray(bufferf))[0] < 16):
		bufferf.append(input_img);
	else:
		inputs = np.reshape(bufferf,[1,16,112,112,3])
		pred = str(labels.iloc[np.argmax(model.predict(inputs)),0])
		bufferf[:-1] = bufferf[1:]; bufferf[-1] = input_img

	# check to see if the frame should be displayed to our screen


	# update the FPS counter
	fps.update()

