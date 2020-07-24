
#                 python3 object_detection_yolo.py --image=bird.jpg
def voter_validate(file_path):   
    import cv2 as cv
    import sys
    import numpy as np
    import os.path
    from scipy import ndimage
    import pytesseract
    from PIL import Image
    import tempfile
    import re
        
    global confidence
    confidence= {}
    # Initialize the parameters
    confThreshold = 0.5  # Confidence threshold
    nmsThreshold = 0.4  # Non-maximum suppression threshold
    
    inpWidth = 416  # 608     #Width of network's input image
    inpHeight = 416  # 608     #Height of network's input image
    
       
    # Load names of classes
    classesFile = "custom_cfg/voter.names"
    
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    
    # Give the configuration and weight files for the model and load the network using them.
    
    modelConfiguration = "custom_cfg/voter.cfg"
    modelWeights = "weights/voter_15000.weights"
    
    
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    
    # Get the names of the output layers
    
    
    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # Draw the predicted bounding box
    
    global identify
    identify=['symbols']
    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
        
        label = '%.2f' % conf
          # Get the label for the class name and its confidence
        if classes:
            assert(classId < len(classes))
            label = '%s:%s' % (classes[classId], label)
            confidence["score"]=int(conf*100)
            global counter 
            counter=label
            identify.append(classes[classId])
           
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
    
        cv.putText(frame, label, (left, top),
                   cv.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
        
    # Remove the bounding boxes with low confidence using non-maxima suppression
    
    
    def postprocess(frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
    
        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            
            for detection in out:
                # if detection[4]>0.001:
                scores = detection[5:]
                classId = np.argmax(scores)
                # if scores[classId]>confThreshold:
    
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        
        
        for i in indices:
    
            # print(i)
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            right = left+width
            bottom = top+height
            
            drawPred(classIds[i], confidences[i], left, top, right, bottom)
        
    # Open the image file
    if not os.path.isfile(file_path):
        print("Input image file ", file_path, " doesn't exist")
        sys.exit(1)
    outputFile = 'result.jpg'
   
    # read image 
    frame=cv.imread(file_path)

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(
        frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.8f ms' % (t * 100000.0 / cv.getTickFrequency())   
    
    if "ECI" in identify:

         #Preprocess the image to ocr   
        def adjust_gamma(image, gamma=1):
            
               invGamma = 1.0 / gamma
               table = np.array([((i / 255.0) ** invGamma) * 255
                  for i in np.arange(0, 256)]).astype("uint8")
            
               return cv.LUT(image, table)
        im = cv.imread(file_path)
         
        width=310
        height=480
        dim= (width,height)
        newImg = cv.resize(im,dim)
        newImg = cv.resize(newImg,(0,0),fx=3,fy=2.08)
        
        
        b,g,r = cv.split(newImg)           # get b,g,r
        rgb_img = cv.merge([r,g,b])     # switch it to rgb
        
        # Denoising
        dst = cv.fastNlMeansDenoisingColored(newImg,None,4,4,3,4)
        
        b,g,r = cv.split(dst)           # get b,g,r
        newImg = cv.merge([r,g,b]) 
        contrast_img = cv.addWeighted(newImg, 1.3, np.zeros(newImg.shape, newImg.dtype), 0, 0)

        gamma = 0.45                              
        adjusted = adjust_gamma(contrast_img, gamma=gamma)
        cv.putText(adjusted, "g={}".format(gamma), (20, 30),cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

        b,g,r = cv.split(adjusted)           # get b,g,r
        rgb_img = cv.merge([r,g,b])     # switch it to rgb
        
        # Denoising
        dst = cv.fastNlMeansDenoisingColored(adjusted,None,6,6,5,6)
        
        b,g,r = cv.split(dst)           # get b,g,r
        adjusted = cv.merge([r,g,b]) 

        img = cv.cvtColor(adjusted, cv.COLOR_BGR2GRAY) 
       
        text = pytesseract.image_to_string(img, lang='eng' , config='--psm 11')
        #To validate the voterid card.
        class Voter_card_Validator:
        
            def __init__(self,text):
                  self.text=text
            
            def is_votercard(self):
                    res=self.text.split()
                    for word in res:
                        if word=='ELECTION' or word=='Election' or word=='ELECTOR':
                            
                            return True
                    
                    return False
        
            def check(self,num):
                sum=0
                check=0
                n=0
                for i in range(7):
                    if check==1:
                        n=int(num[i])
                        n=n*2
                        check=0
                    else:
                        n=int(num[i])
                        check=1
                    if n>=10:
                        sum=sum+(n-9)
                    else:
                        sum=sum+n
                return sum%10==0
        
            def is_valid(self):
                if (self.is_votercard()):
                    res=self.text.split()
                    all = re.findall(r"[\S]{2,3}/[\d]{2}/[\d]{2}/[\d]{6}", self.text)
                    p=0
                    if all:
                        for i in all:
                            confidence["EPIC Number"]=i
                            p=1
                            
                   #To find the EPIC number. 
                    if p==0:
                        wordindex=-1
                        for word in res:
                            c=0
                            for i in word:
                                c=c+1
                            if c==10:
                                part1=word[0:3]
                                part2=word[3::]
                                part1 = part1.replace("8", "B")
                                part1 = part1.replace("0", "D")
                                part1 = part1.replace("6", "G")
                                part1 = part1.replace("1", "I")
                                part1 = part1.replace("0", "O")
                                part1 = part1.replace("5", "S")
                                part2 = part2.replace("B", "8")
                                part2 = part2.replace("D", "0")
                                part2 = part2.replace("G", "6")
                                part2 = part2.replace("I", "1")
                                part2 = part2.replace("O", "0")
                                part2 = part2.replace("S", "5")
                                checkword=part1+part2
                                if(checkword[0:3].isalpha()):
                                    if(checkword[3::].isdigit()):
                                        if self.check(checkword[3::]):
                                            wordindex=res.index(word)
                                            confidence["result"]="Valid voter Card"
                                            confidence["EPIC Number"]= checkword
                                            break
                                        else:
                                            confidence["result"]="Not a Valid voter Card"
                            if c==3:
                                index=res.index(word)
                                wo=res[index]+res[index+1]
                                c=0
                                for i in wo:
                                    c=c+1
                                if c==10:
                                    part1=wo[0:3]
                                    part2=wo[3::]
                                    part1 = part1.replace("8", "B")
                                    part1 = part1.replace("0", "D")
                                    part1 = part1.replace("6", "G")
                                    part1 = part1.replace("1", "I")
                                    part1 = part1.replace("0", "O")
                                    part1 = part1.replace("5", "S")
                                    part2 = part2.replace("B", "8")
                                    part2 = part2.replace("D", "0")
                                    part2 = part2.replace("G", "6")
                                    part2 = part2.replace("I", "1")
                                    part2 = part2.replace("O", "0")
                                    part2 = part2.replace("S", "5")
                                    checkwo=part1+part2
                                    if(checkwo[0:3].isalpha()):
                                        if(checkwo[3::].isdigit()):
                                            if self.check(checkwo[3::]):
                                                wordindex=res.index(word)+1
                                                confidence["result"]="Not a Valid voter Card"
                                                confidence["EPIC Number"]= checkwo
                                                break
                                            else:
                                                confidence["result"]="Valid voter Card"
        
               
                    
                    names_index=[]
                    index=0
                    for word in res:
                        if word=='NAME' or word=='Name' or word== 'Narne':
                            names_index.append(index+1)
                        index=index+1
        
                    if len(names_index)==2:
                        flag=1
                        for i in names_index:
                            while len(res[i])<=3:
                                    i=i+1
                            j=i+1
                            while j<len(res) and len(res[j])<3:
                                j=j+1
                            if j<len(res) and res[j].isalpha():
                                if flag:
                                    confidence["Name"]= res[i]+ ' ' + res[j] 
                                    flag=0
                                    name=1
                                else:
                                    confidence["Father's Name"]= res[i]+ ' ' + res[j]
                                    return
                            if j>len(res):
                                if flag:
                                    confidence["Name"]= res[i]
                                    flag=0
                                    name=1
                                else:
                                    confidence["Father's Name"]= res[i]
                                    return
                    if len(names_index)==0:
                        flag=1
                        for m in res:
                            if res.index(m)>wordindex and m.isupper() and flag and len(m)>2 and m.isalpha():
                                i=res.index(m)
                                a=res.index(m)+1
                                while a<len(res) and len(res[a])<3:
                                    a=a+1
                                if a<len(res) and res[a].isalpha():
                                    confidence["Name"]= m+' '+res[a]
                                else:
                                    confidence["Name"]= m
                                flag=0
                            elif res.index(m)>wordindex and m.isupper() and len(m)>2 and m.isalpha():
                                i=res.index(m)
                                a=res.index(m)+1
                                while a<len(res) and len(res[a])<3 :
                                    a=a+1
                                if a<len(res) and res[a].isalpha():
                                    confidence["Father's Name"]= m+' '+res[a]
                                else:
                                    confidence["Father's Name"]= m
                                break
                    
                    flag=1
                    if len(names_index)==1:
                        while len(res[names_index[0]])<3:
                            names_index[0]=names_index[0]+1
                        j=names_index[0]+1
                   
                        
                        for m in res:
                            if res.index(m)>names_index[0]+3 and m.isupper() and len(m)>2 and m.isalpha():
                                if j<len(res) and res[j].isalpha():
                                    confidence["Name"]= res[names_index[0]]+' ' + res[j]
                                else:
                                    confidence["Name"]= res[names_index[0]]
                                a=res.index(m)+1
    
                                if a<len(res) and res[a].isalpha() and len(res[a])>=3:
                                    confidence["Father's Name"]= m+' '+res[a]
                                else:
                                    confidence["Father's Name"]= m
                                flag=0
                                break
                        if flag:
                            for m in res:
                                if res.index(m)>wordindex and m.isupper() and len(m)>2 and m.isalpha():
                                    a=res.index(m)+1
                                    while a<len(res) and len(res[a])<3 :
                                        a=a+1
                                    if a<len(res) and res[a].isalpha() and len(res[a])>=3:
                                        confidence["Name"]= m+' '+res[a]
                                    else:
                                        confidence["Name"]= m
                                    if j<len(res) and res[j].isalpha():
                                        confidence["Father's Name"]=res[names_index[0]]+' ' + res[j]
                                    else:
                                         confidence["Father's Name"]= res[names_index[0]]
                                    return
                                    
        
        
        
        def main():
             dlv=Voter_card_Validator(text)
             dlv.is_valid()
             
        main()
    
    
    
    else:
        confidence["result"]="Not a Voter Card"
        confidence["Name"]="NULL"
        confidence["Father's Name"]="NULL"
        confidence["EPIC Number"]="NULL"
    if 'EPIC Number' in confidence:
        return confidence
    else:
        confidence["result"]="Valid Voter Card"
        confidence["Name"]="NULL"
        confidence["Father's Name"]="NULL"
        confidence["EPIC Number"]="NULL"
        return confidence
