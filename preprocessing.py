import cv2 
import numpy as np

#(321, 1048), (333, 0), (1620, 0), (1636, 1048)]
#[(188, 1048), (168, 0), (1600, 0), (1644, 1048)]


first_mask = np.zeros((1200, 1920, 3), dtype=int)
first_mask[:, 320:1636]=(1, 1, 1)
second_mask = np.zeros((1200, 1920, 3), dtype=int)
second_mask[:, 168:1644] = (1, 1, 1)
def preprocess(path):
    imm = cv2.imread(path)
    if(path.endswith("first.png")):
        imm[:, 0:320]=(0,0,0)
        imm[:, 1636:]=(0,0,0)
    elif(path.endswith("second.png")):
        imm[:, 0:168]=(0,0,0)
        imm[:, 1644:]=(0,0,0)
    imm = cv2.resize(imm,(224,224))    
    imm = cv2.cvtColor(cv2.GaussianBlur(imm, (5, 5), 0), cv2.COLOR_BGR2YCrCb)
    imm[:,:,0] = cv2.equalizeHist(imm[:,:,0])
    imm = cv2.cvtColor(imm, cv2.COLOR_YCrCb2BGR)
    return imm

