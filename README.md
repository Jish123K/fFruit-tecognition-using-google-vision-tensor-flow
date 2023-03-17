# fFruit-tecognition-using-google-vision-tensor-flow
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
