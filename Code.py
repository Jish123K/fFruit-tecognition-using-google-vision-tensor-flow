import io

import os

import cv2

import numpy as np

from datetime import datetime

import tensorflow as tf

from google.cloud import vision_v1p3beta1 as vision

# Set up Google authentication client key

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client_key.json'

# Set up path to image folder

SOURCE_PATH = "E:/temp_uploads/Photos/Fruit/"

# Set up list of food types

FOOD_TYPE = 'Fruit'  # 'Vegetable'

# Set up object detection model

MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'

PATH_TO_MODEL = os.path.join('object_detection_models', MODEL_NAME, 'saved_model')

DETECTION_THRESHOLD = 0.5

# Load label map

PATH_TO_LABELS = os.path.join('object_detection_models', 'mscoco_label_map.pbtxt')

with open(PATH_TO_LABELS, 'r') as f:

    labels = f.read().split('\n')

category_index = {}

for line in labels:

    if 'id:' in line:

        id = int(line.split(':')[1])

    if 'name:' in line:

        name = line.split(':')[1].replace("'", "").strip()

        category_index[id] = {'name': name}

# Load list of known food names

def load_food_names(food_type):

    names = [line.rstrip('\n').lower() for line in open(food_type + '.dict')]

    return names

# Use Google Cloud Vision API to label image

def label_image(img_path):

    client = vision.ImageAnnotatorClient()

    with io.open(img_path, 'rb') as image_file:

        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.label_detection(image=image)

    labels = response.label_annotations

    return [label.description.lower() for label in labels]

# Use object detection model to detect food in image

def detect_food(img_path, detection_model):

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = tf.convert_to_tensor(img)

    img_tensor = img_tensor[tf.newaxis, ...]

    detections = detection_model(img_tensor)

    detections = detections.numpy().squeeze()

    classes = detections['detection_classes']

    scores = detections['detection_scores']

    boxes = detections['detection_boxes']

    # Filter out low-confidence detections

    mask = scores >= DETECTION_THRESHOLD

    classes = classes[mask]

    scores = scores[mask]

    boxes = boxes[mask]

    # Convert box coordinates to pixel values

    height, width, _ = img.shape

    boxes[:, 0] = boxes[:, 0] * height

    boxes[:, 1] = boxes[:, 1] * width

    boxes[:, 2] = boxes[:, 2] * height

    boxes[:, 3] = boxes[:, 3] * width

    boxes = boxes.astype(np.int32)

    return classes, scores, boxes

# Draw bounding boxes around detected food in image
def draw_boxes(img_path, classes, scores, boxes):
    img = cv2.imread(img_path)
    for i in range(len(classes)):
        class_id = int(classes[i])
        class_name = category_index[class_id]['name']
        score = scores[i]
        ymin, xmin, ymax, xmax = boxes[i]
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, "{}: {:.2f}".format(class_name, score), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


