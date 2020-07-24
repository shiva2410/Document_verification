from __future__ import print_function
import cv2
import numpy as np
import pytesseract
from PIL import Image
import gpyocr
import argparse
import os
import tempfile
import sys
import sys
from aadhaar_card.aadhaar_object import detect_features  
global confidence
confidence={
    "result":"Invalid Aadhaar Card",
    "score":"0%",
    "Name":"Null",
    "Gender":"Null",
    "DOB":"Null",
    "Aadhaar Number":"Null"
}
import methods


def recognize(imgname,name):
    global confidence
    processed_img_file=methods.Gaussian_blur(imgname)
    features_list,feature_img=detect_features(processed_img_file)
    print(len(features_list))
    if len(features_list)<2:  
        confidence['result']="Invalid Aadhaar Card"
        confidence['score']="0%"
        confidence['Name']='NULL'
        confidence['Gender']='NULL'
        confidence['DOB']='NULL'
        confidence['Aadhaar Number']='NULL'
        return(confidence)  
    else:
        customer_name=name
        img=cv2.imread(processed_img_file)
        gcp_text=gpyocr.google_vision_ocr(img, langs=['en'])
        recognized_text = ''.join(gcp_text[0])
        confidence=methods.printdata(recognized_text,customer_name.upper())
        return confidence
