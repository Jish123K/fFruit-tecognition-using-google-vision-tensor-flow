# fFruit-tecognition-using-google-vision-tensor-flow
The code is designed to perform two main tasks: object detection and label recognition of food in images. Here's a breakdown of how the code works:

Import required libraries: The code starts by importing the necessary libraries including OpenCV, NumPy, TensorFlow, Google Cloud Vision API, and datetime.

Set up authentication: The code sets up Google authentication client key by setting the 'GOOGLE_APPLICATION_CREDENTIALS' environment variable to the location of the client key file.

Set up paths and parameters: The code sets up the path to the image folder, food type (fruit or vegetable), object detection model name, path to the model, and detection threshold. It also loads the label map and the list of known food names.

Define functions: The code defines several functions for labeling images, detecting food in images, and drawing bounding boxes around detected food.

Main function: The main function of the code reads the images from the specified image folder, performs object detection and label recognition on each image, and saves the labeled images in a new folder.
computer vision
Here's how the main function works:

It loops through all the image files in the specified image folder.
For each image file, it first performs label recognition using the Google Cloud Vision API and returns a list of labels.
It then filters out the labels that are not food-related by comparing them with the list of known food names.
If there are no food-related labels, the image is skipped.
If there are food-related labels, the function uses the object detection model to detect the food in the image and returns the class IDs, scores, and bounding box coordinates.
It then draws bounding boxes around the detected food and saves the labeled image in a new folder.
Requirements:
The code has the following requirements:

TensorFlow object detection API: The code uses the TensorFlow object detection API to perform object detection on images. Therefore, you need to have the TensorFlow object detection API installed and set up.
Google Cloud Vision API: The code uses the Google Cloud Vision API to perform label recognition on images. Therefore, you need to have a Google Cloud Platform account and set up the Google Cloud Vision API.
Object detection model: The code uses the 'ssd_mobilenet_v2_coco_2018_03_29' object detection model. You can use a different object detection model if you want, but you will need to update the code accordingly.
Label map and list of known food names: The code requires a label map and a list of known food names to filter out non-food labels. You can create your own label map and food name list, or use the ones provided in the code.
Tenchologies used:
1.deep learning
2.Tensor flow
3.Python
4.Artifical intelligence 
