import cv2
import os
import os.path
import json
import sys
import re
import csv
import dateutil.parser as dparser
from PIL import Image
import os
import tempfile   
import time
import recognize_doc
confidence=recognize_doc.confidence
 
global flag
flag=1

def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False,suffix='.png')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename

def Gaussian_blur(imgname):
    src=cv2.imread(imgname)
    processed_img=cv2.GaussianBlur(src,(3,3),0)
    temp_file = tempfile.NamedTemporaryFile(delete=False,   suffix='.jpg')
    temp_filename = temp_file.name
    cv2.imwrite(temp_filename,processed_img)
    return temp_filename

def downscale(orig_im, downscaled_height):
    scale = orig_im.shape[0] / downscaled_height
    im = resize(orig_im, height=int(downscaled_height))
    return im, scale

def input_name():

    master = tkinter.Tk()
    tkinter.Label(master,text="Name of the customer").grid(row=0)
    e1 = tkinter.Entry(master)
    e1.grid(row=0, column=1)
    tkinter.Button(master,text='Submit',command=master.quit).grid(row=2, column=1, sticky=tkinter.W,pady=4)
    tkinter.mainloop()
    return e1.get()
    



mult = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
perm = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def Validate_aadhaar(aadharNum):
    global flag
    aadharNum=aadharNum.replace(" ","")
    try:
        if (len(aadharNum) == 12 and aadharNum.isdigit()):
            i = len(aadharNum)
            j = 0
            x = 0

            while i > 0:
                i -= 1
                x = mult[x][perm[(j % 8)][int(aadharNum[i])]]
                j += 1
            if x == 0:
                return 1
            else:
                return 0
        else:
            return 0

    except ValueError:
        return 0
    except IndexError:
        return 0



def printdata(extracted_text,customer_name):
    global flag

    text=extracted_text
    # Initializing data variable
    name = ''
    gender = None
    dob = None
    uid=[]
    uid = None
    yearline = []
    genline = []
    nameline = []
    text1 = []
    text2 = []
    genderStr = '(Female|Male|emale|male|ale|FEMALE|MALE|EMALE)$'


    # Searching for Year of Birth
    lines = text
    # print (lines)
    for wordlist in lines.split('\n'):
        xx = wordlist.split()
        if [w for w in xx if re.search('(Year|Birth|irth|YoB|YOB:|DOB:|DOB)$', w)]:
            yearline = wordlist
            break
        else:
            text1.append(wordlist)
    try:
        text2 = text.split(yearline, 1)[1]
    except Exception:
        pass

    try:
        yearline = re.split('Year|Birth|irth|YoB|YOB:|DOB:|DOB|of|Birth :|Birth:|:| :', yearline)[1:]
        yearline = ''.join(str(e) for e in yearline)
        if yearline:
            dob = dparser.parse(yearline, fuzzy=True).year
            dob=yearline
    except Exception:
        pass

    # Searching for Gender
    try:
        for wordlist in lines.split('\n'):
            xx = wordlist.split()
            if [w for w in xx if re.search(genderStr, w)]:
                genline = wordlist
                break
        if 'Male' in genline or 'MALE' in genline:
            gender = "Male"
        if 'Female' in genline or 'FEMALE' in genline:
            gender = "Female"

        text2 = text.split(genline, 1)[1]
    except Exception:
        pass

    try:
        name_con=map(str.upper,text1)
        if customer_name.upper() in name_con:
            name=customer_name
            flag=1
        else:
            flag=0
            confidence['result']="Name does not match with database"
            confidence['score']="50%"
            confidence['Name']='NULL'
            confidence['Gender']='NULL'
            confidence['DOB']='NULL'
            confidence['Aadhaar Number']='NULL'
            return(confidence) 
    except Exception:
        pass

# Searching for UID
    uid = set()
    try:
        newlist = []
        for j in text2.split('\n'):
            newlist.append(j)
        newlist = list(filter(lambda x: len(x)==14, newlist))
        for n in newlist:
            uid.add(n)

    except Exception:
        pass

    data = {}
    data['Name'] = name.upper()
    data['Gender'] = gender
    data['Birth year'] = dob
    uid=list(uid)
    if len(list(uid)) >= 1:
        var = Validate_aadhaar(uid[0])
        if var==1 and flag==1:
            data['Uid']=uid[0]
            flag=1
        elif var==0 and flag==1:
            flag=0
            confidence['result']="Invalid Aadhaar Number according to Verhoef Algorithm"
            confidence['score']="60%"
            confidence['Name']='NULL'
            confidence['Gender']='NULL'
            confidence['DOB']='NULL'
            confidence['Aadhaar Number']='NULL'
            return(confidence)
  
    if flag==1:
        confidence['result']="Valid Aadhaar Card"
        confidence['score']="90%"
        confidence['Name']=data['Name']
        confidence['Gender']=data['Gender']
        confidence['DOB']=data['Birth year']
        confidence['Aadhaar Number']=data['Uid']
        return(confidence)
    else:
        return("flag is "+str(flag))