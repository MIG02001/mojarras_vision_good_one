######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/2/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util


class GeometricCorrection:
    def __init__(self):
        self.x_1 = 1920
        self.x_2 = 1280
        self.y_1 = 1080
        self.y_2 = 720
        
    @staticmethod
    def correction_x(self, x_before) -> int:
        x_before * self.x_2 / self.x_1
        
    @staticmethod
    def correction_y(self, y_before) -> int:
        y_before * self.y_2 / self.y_1
        
        
        
# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--video', help='Name of the video file',
                    default='test.mp4')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

#Variables a enviar
R1 = 1
R2 = 1
R3 = 1
R4 = 1

#Coordenadas R1 para condición SE VA AL NUEVO
roi_xmin1 = 815
roi_ymin1=  630
roi_xmax1 = 970
roi_ymax1 = 740
roi_xmin2 = 670
roi_ymin2=  635
roi_xmax2 = 830
roi_ymax2 = 760
roi_xmin3 = 485
roi_ymin3=  650
roi_xmax3 = 680
roi_ymax3 = 765
roi_xmin4 = 320
roi_ymin4=  665
roi_xmax4 = 485
roi_ymax4 = 775
roi_xmin5 = 173
roi_ymin5=  680
roi_xmax5 = 325
roi_ymax5 = 780
roi_xmin6 = 25
roi_ymin6=  690
roi_xmax6 = 176
roi_ymax6 = 790

#R2
r2_xmin1 = 865
r2_ymin1 = 615
r2_xmax1 = 1140
r2_ymax1 = 700

r2_xmin2 = 900
r2_ymin2 = 708
r2_xmax2 = 1210
r2_ymax2 = 795

r2_xmin3 = 980
r2_ymin3 = 790
r2_xmax3 = 1260
r2_ymax3 = 925

r2_xmin4 = 950
r2_ymin4 = 930
r2_xmax4 = 1400
r2_ymax4 = 1100

#R3
r3_xmin1 = 1200
r3_ymin1 = 520
r3_xmax1 = 1300
r3_ymax1 = 650
r3_xmin2 = 1320
r3_ymin2 = 505
r3_xmax2 = 1395
r3_ymax2 = 637
r3_xmin3 = 1410
r3_ymin3 = 490
r3_xmax3 = 1500
r3_ymax3 = 630
r3_xmin4 = 1520
r3_ymin4 = 490
r3_xmax4 = 1610
r3_ymax4 = 625
r3_xmin5 = 1620
r3_ymin5 = 485
r3_xmax5 = 1720
r3_ymax5 = 620

#R4
r4_xmin1 = 810
r4_ymin1 = 490
r4_xmax1 = 1015
r4_ymax1 = 560
r4_xmin2 = 790
r4_ymin2 = 440
r4_xmax2 = 980
r4_ymax2 = 510
r4_xmin3 = 765
r4_ymin3 = 400
r4_xmax3 = 970
r4_ymax3 = 430
r4_xmin4 = 600
r4_ymin4 = 375
r4_xmax4 = 750
r4_ymax4 = 430

#Conficionales en función
#Condicionales de R1
def areasR1(ymin,xmin,ymax,xmax):
    global R1   
    if roi_xmin6 <= xmin <= roi_xmax6 and roi_ymin6 <= ymin <= roi_ymax6:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R1 = 6
    elif roi_xmin5 <= xmin <= roi_xmax5 and roi_ymin5 <= ymin <= roi_ymax5:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R1 = 5
    elif roi_xmin4 <= xmin <= roi_xmax4 and roi_ymin4 <= ymin <= roi_ymax4:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R1 = 4
    elif roi_xmin3 <= xmin <= roi_xmax3 and roi_ymin3 <= ymin <= roi_ymax3:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R1 = 3
    elif roi_xmin2 <= xmin <= roi_xmax2 and roi_ymin2 <= ymin <= roi_ymax2:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R1 = 2
    elif roi_xmin1 <= xmin <= roi_xmax1 and roi_ymin1 <= ymin <= roi_ymax1:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R1 = 1
    return R1

#Condicionales R2
def areasR2(ymin,xmin,ymax,xmax):
    global R2
    if r2_xmin4 <= xmin <= r2_xmax4 and r2_ymin4 <= ymin <= r2_ymax4:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R2 = 4
    elif r2_xmin3 <= xmin <= r2_xmax3 and r2_ymin3 <= ymin <= r2_ymax3:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R2 = 3
    elif r2_xmin2 <= xmin <= r2_xmax2 and r2_ymin2 <= ymin <= r2_ymax2:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R2 = 2
    elif r2_xmin1 <= xmin <= r2_xmax1 and r2_ymin1 <= ymin <= r2_ymax1:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R2 = 1
    return R2

#R3 conditionals
def areasR3(ymin,xmin,ymax,xmax):
    global R3
    if r3_xmin5 <= xmin <= r3_xmax5 and r3_ymin5 <= ymin <= r3_ymax5:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R3 = 5
    elif r3_xmin4 <= xmin <= r3_xmax4 and r3_ymin4 <= ymin <= r3_ymax4:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R3 = 4
    elif r3_xmin3 <= xmin <= r3_xmax3 and r3_ymin3 <= ymin <= r3_ymax3:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R3 = 3
    elif r3_xmin2 <= xmin <= r3_xmax2 and r3_ymin2 <= ymin <= r3_ymax2:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R3 = 2
    elif r3_xmin1 <= xmin <= r3_xmax1 and r3_ymin1 <= ymin <= r3_ymax1:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R3 = 1
    return R3

#R4 conditionals
def areasR4(ymin,xmin,ymax,xmax):
    global R4
    if r4_xmin4 <= xmin <= r4_xmax4 and r4_ymin4 <= ymin <= r4_ymax4:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R4 = 4
    elif r4_xmin3 <= xmin <= r4_xmax3 and r4_ymin3 <= ymin <= r4_ymax3:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R4 = 3
    elif r4_xmin2 <= xmin <= r4_xmax2 and r4_ymin2 <= ymin <= r4_ymax2:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R4 = 2
    elif r4_xmin1 <= xmin <= r4_xmax1 and r4_ymin1 <= ymin <= r4_ymax1:
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        # Draw label
        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        R4 = 1
    return R4


# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

while(video.isOpened()):

    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
      print('Reached the end of the video!')
      break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > .05) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            R1 = areasR1(ymin,xmin,ymax,xmax)
            R2 = areasR2(ymin,xmin,ymax,xmax)
            R3 = areasR3(ymin,xmin,ymax,xmax)
            R4 = areasR4(ymin,xmin,ymax,xmax)
            
                

    # All the results have been drawn on the frame, so it's time to display it.
    #R1 rectangles
    cv2.rectangle(frame,(roi_xmin1,roi_ymin1),(roi_xmax1,roi_ymax1),(0,255,255),1)
    cv2.rectangle(frame, (roi_xmin2,roi_ymin2),(roi_xmax2,roi_ymax2),(0,0,255),1)
    cv2.rectangle(frame, (roi_xmin3,roi_ymin3),(roi_xmax3,roi_ymax3),(0,0,0),1)
    cv2.rectangle(frame, (roi_xmin4,roi_ymin4),(roi_xmax4,roi_ymax4),(255,0,0),1)
    cv2.rectangle(frame, (roi_xmin5,roi_ymin5),(roi_xmax5,roi_ymax5),(255,150,0),1)
    cv2.rectangle(frame, (roi_xmin6,roi_ymin6),(roi_xmax6,roi_ymax6),(180,0,255),1)
    
    #R2 rectangles
    cv2.rectangle(frame,(r2_xmin1,r2_ymin1),(r2_xmax1,r2_ymax1),(0,255,255),1)
    cv2.rectangle(frame, (r2_xmin2,r2_ymin2),(r2_xmax2,r2_ymax2),(0,0,255),1)
    cv2.rectangle(frame, (r2_xmin3,r2_ymin3),(r2_xmax3,r2_ymax3),(0,0,0),1)
    cv2.rectangle(frame, (r2_xmin4,r2_ymin4),(r2_xmax4,r2_ymax4),(255,0,0),1)
    
    #R3 rectrangles
    cv2.rectangle(frame,(r3_xmin1,r3_ymin1),(r3_xmax1,r3_ymax1),(0,255,255),1)
    cv2.rectangle(frame, (r3_xmin2,r3_ymin2),(r3_xmax2,r3_ymax2),(0,0,255),1)
    cv2.rectangle(frame, (r3_xmin3,r3_ymin3),(r3_xmax3,r3_ymax3),(0,0,0),1)
    cv2.rectangle(frame, (r3_xmin4,r3_ymin4),(r3_xmax4,r3_ymax4),(255,0,0),1)
    cv2.rectangle(frame, (r3_xmin5,r3_ymin5),(r3_xmax5,r3_ymax5),(180,0,255),1)
    
    #R4 rectangles
    cv2.rectangle(frame, (r4_xmin1,r4_ymin1),(r4_xmax1,r4_ymax1),(0,255,255),1)
    cv2.rectangle(frame, (r4_xmin2,r4_ymin2),(r4_xmax2,r4_ymax2),(0,0,255),1)
    cv2.rectangle(frame, (r4_xmin3,r4_ymin3),(r4_xmax3,r4_ymax3),(0,0,0),1)
    cv2.rectangle(frame, (r4_xmin4,r4_ymin4),(r4_xmax4,r4_ymax4),(180,0,255),1)
    
    
    #Printing results
    R1 = str(R1)
    R2 = str(R2)
    R3 = str(R3)
    R4 = str(R4)
    print(f"current Rs" + R1 + "  " + R2 + "  " + R3 + "  " + R4)
        
    #Frame conf
    frame_resized = cv2.resize(frame, None, interpolation = cv2.INTER_CUBIC, fx = 0.8, fy = 0.78) 
    cv2.imshow('Object detector', frame_resized)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    

        
# Clean up
video.release()
cv2.destroyAllWindows()
     
