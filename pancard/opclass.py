
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import os
import sys
import numpy as np

class Classifier:
    def __init__(self):
        self.model=None
        self.img=None
        self.x=0
        self.class_probabilities=0
        self.pred_class=""
        self.MODEL_LOADED=False
        self.FAULTY_IMAGE=True
        
    def init_model(self,weights_file='weights-best-improvement-09-0.94.hdf5'):
        """The Model file is loaded."""
        print("here in model function")
        try:
            print("here model load")
            if(self.MODEL_LOADED==False):
                print("loading")
                self.model = load_model(weights_file)
                self.MODEL_LOADED=True
                return 0
        except:
            sys.exit("Unable to find weights file.")
            return 1
    
    def check_faulty_image(self,image):
        """Image is first verified at the given path and is checked
        for faults."""
        try:        
            fh=open(image,'r')
            self.img = cv2.imread(image, 0)
            if cv2.countNonZero(self.img) == 0:
                self.FAULTY_IMAGE=True
            else:
                self.img=cv2.resize(self.img,(150,150))
                rows,cols = self.img.shape
                for i in range(rows):
                    for j in range(cols):
                        k = self.img[i,j]
                sdev=np.std(self.img)
                #print(sdev)
                #vals=np.mean(self.img)
                #vals=abs(vals-k)
                #print(vals)
                #if vals==0.0:
                if sdev<15.0:
                    self.FAULTY_IMAGE=True
                else:
                    self.FAULTY_IMAGE=False
        except:
            error="File: \'"+image+"\' not found at specified path."
            sys.exit(error) 
            pass
           
    def get_prediction(self,image,confidence=5e-01):
        """Prediction is made for given Image."""
        if self.MODEL_LOADED==False:
            sys.exit("Weights have not been loaded.")
            return 1
        else:
            self.check_faulty_image(image)
            if self.FAULTY_IMAGE==True:
                #sys.exit("Faulty image found, cannot process.")
                self.pred_class='others'
                return self.pred_class.upper()
            
            else:
                self.img=load_img(image)
                self.x=img_to_array(self.img)
                self.x=self.x/255
                self.x=cv2.resize(self.x,(150,150))
                self.x = self.x.reshape((1,) + self.x.shape)
                self.class_probabilities = self.model.predict_proba(self.x)
                if self.class_probabilities[0][1]>confidence:
                    self.pred_class='others'
                else:
                    if self.class_probabilities[0][0]>self.class_probabilities[0][2]:
                        self.pred_class='others'
                    else:
                        self.pred_class='pan'
                    
            return self.pred_class.upper()
