#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


# In[26]:


class EncodeFaces:
    
    def __init__(self, dataset_path=None):
        # grab the paths to the input images in our dataset
        if dataset_path is not None:
            print("[INFO] quantifying faces...")
            self.imagePaths = list(paths.list_images(dataset_path))
        # initialize the list of known encodings and known names
        self.knownEncodings = []
        self.knownNames = []
        
    def encodeFaces(self, detection_method):
        # loop over the image paths
        for (i, imagePath) in enumerate(self.imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1,len(self.imagePaths)))
            name = imagePath.split(os.path.sep)[-2]

            # load the input image and convert it from RGB (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            boxes = face_recognition.face_locations(rgb, model=detection_method)

            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)

            # loop over the encodings
            for encoding in encodings:
                # add each encoding + name to our set of known names and
                # encodings
                self.knownEncodings.append(encoding)
                self.knownNames.append(name)
                
    def dumpEncoding(self, dump_file_path):
        # dump the facial encodings + names to disk
        print("[INFO] serializing encodings...")
        data = {"encodings": self.knownEncodings, "names": self.knownNames}
        f = open(dump_file_path, "wb")
        f.write(pickle.dumps(data))
        f.close()


# In[30]:


# obj = EncodeFaces('/home/dai/Documents/pgdai/project/project Data/temp_test_data')
# obj.encodeFaces(detection_method='cnn')
# obj.dumpEncoding('/home/dai/Documents/pgdai/project/project Data/cnn_encoding.pickle')


# In[32]:


# python face_encoding.py --dataset dataset --encodings encodings.pickle
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, 
                help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True, 
                help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn", 
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())
if all(v is not None for v in [args["dataset"], args["encodings"], args["detection_method"]]):
    obj = EncodeFaces('/home/dai/Documents/pgdai/project/project Data/temp_test_data')
    obj.encodeFaces(detection_method='cnn')
    obj.dumpEncoding('/home/dai/Documents/pgdai/project/project Data/cnn_encoding.pickle')


# In[ ]:




