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

class Point:
    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x = x
        self.y = y

class Area:
    def __init__(self, minimum: Point, maximum: Point) -> None:
        self.minimum = minimum
        self.maximum = maximum

    def isAreaInsideOf(self, minimum: Point, maximum: Point) -> bool:
        if self.minimum.x <= minimum.x and self.minimum.y <= minimum.y and self.maximum.x >= maximum.x and self.maximum.y >= maximum.y:
            return True
        return False

class Drawer:
    def __init__(self, frame):
        self.frame = frame
        self.color = (0, 0, 255)

    def draw_rectangle(self, frame, minimum: Point, maximum: Point):
        cv2.rectangle(frame, (minimum.x, minimum.y), (maximum.x, maximum.y), self.color,1)
        
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

#! Coordenadas R1
r1_min1 = Point(815, 630)
r1_max1 = Point(970, 740)

r1_min2 = Point(670, 635)
r1_max2 = Point(830, 760)

r1_min3 = Point(485, 650)
r1_max3 = Point(680, 765)

r1_min4 = Point(320, 665)
r1_max4 = Point(485, 775)

r1_min5 = Point(173, 680)
r1_max5 = Point(325, 780)

r1_min6 = Point(25, 690)
r1_max6 = Point(176, 790)

#! Coordenadas R2
r2_min1 = Point(865, 615)
r2_max1 = Point(1140, 700)

r2_min2 = Point(900, 708)
r2_max2 = Point(1210, 795)

r2_min3 = Point(980, 790)
r2_max3 = Point(1260, 925)

r2_min4 = Point(950, 930)
r2_max4 = Point(1400, 1100)

#! Coordenadas R3
r3_min1 = Point(1200, 520)
r3_max1 = Point(1300, 650)

r3_min2 = Point(1320, 505)
r3_max2 = Point(1395, 637)

r3_min3 = Point(1410, 490)
r3_max3 = Point(1500, 630)

r3_min4 = Point(1520, 490)
r3_max4 = Point(1610, 625)

r3_min5 = Point(1620, 485)
r3_max5 = Point(1720, 620)

#! Coordenadas R4
r4_min1 = Point(810, 490)
r4_max1 = Point(1015, 560)

r4_min2 = Point(790, 440)
r4_max2 = Point(980, 510)

r4_min3 = Point(765, 400)
r4_max3 = Point(970, 430)

r4_min4 = Point(600, 375)
r4_max4 = Point(750, 430)

#! Areas
areas_r1 = [Area(r1_min1, r1_max1), Area(r1_min2, r1_max2), Area(r1_min3, r1_max3), Area(r1_min4, r1_max4), Area(r1_min5, r1_max5), Area(r1_min6, r1_max6)]
areas_r2 = [Area(r2_min1, r2_max1), Area(r2_min2, r2_max2), Area(r2_min3, r2_max3), Area(r2_min4, r2_max4)]
areas_r3 = [Area(r3_min1, r3_max1), Area(r3_min2, r3_max2), Area(r3_min3, r3_max3), Area(r3_min4, r3_max4), Area(r3_min5, r3_max5)]
areas_r4 = [Area(r4_min1, r4_max1), Area(r4_min2, r4_max2), Area(r4_min3, r4_max3), Area(r4_min4, r4_max4)]

class GeometricCalculation:
    def __init__(self, minimumPoint: Point, maximumPoint: Point) -> None:
        self.minimumPoint = minimumPoint
        self.maximumPoint = maximumPoint
        self.area = Area(self.minimumPoint, self.maximumPoint)

    def calculate_number_of_cars(self, areas: list) -> int:
        maximum_number = 0
        for idx, area in enumerate(areas):
            isCarInside = self.area.isAreaInsideOf(area.minimum, area.maximum)
            if isCarInside:
                possible_maximum = idx + 1
                if possible_maximum > maximum_number:
                    maximum_number = possible_maximum
        return maximum_number

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

            area_calculator = GeometricCalculation(Point(xmin, ymin), Point(xmax, ymax))

            R1 = area_calculator.calculate_number_of_cars(areas_r1)
            R2 = area_calculator.calculate_number_of_cars(areas_r2)
            R3 = area_calculator.calculate_number_of_cars(areas_r3)
            R4 = area_calculator.calculate_number_of_cars(areas_r4)

            print(f"current Rs" + R1 + "  " + R2 + "  " + R3 + "  " + R4)
            
    drawer = Drawer(frame=frame)

    #! R1 Rectangles
    drawer.draw_rectangle(r1_min1, r1_max1)  
    drawer.draw_rectangle(r1_min2, r1_max2)
    drawer.draw_rectangle(r1_min3, r1_max3)
    drawer.draw_rectangle(r1_min4, r1_max4)
    drawer.draw_rectangle(r1_min5, r1_max5)
    drawer.draw_rectangle(r1_min6, r1_max6)

    #! R2 Rectangles
    drawer.draw_rectangle(r2_min1, r2_max1)
    drawer.draw_rectangle(r2_min2, r2_max2)
    drawer.draw_rectangle(r2_min3, r2_max3)
    drawer.draw_rectangle(r2_min4, r2_max4)

    #! R3 Rectrangles
    drawer.draw_rectangle(r3_min1, r3_max1)
    drawer.draw_rectangle(r3_min2, r3_max2)
    drawer.draw_rectangle(r3_min3, r3_max3)
    drawer.draw_rectangle(r3_min4, r3_max4)
    
    #! R4 Rectangles
    drawer.draw_rectangle(r4_min1, r4_max1)
    drawer.draw_rectangle(r4_min2, r4_max2)
    drawer.draw_rectangle(r4_min3, r4_max3)
    drawer.draw_rectangle(r4_min4, r4_max4)
        
    #Frame conf
    frame_resized = cv2.resize(frame, None, interpolation = cv2.INTER_CUBIC, fx = 0.8, fy = 0.78) 
    cv2.imshow('Object detector', frame_resized)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()


