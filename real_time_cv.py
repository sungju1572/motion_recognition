import cv2
import time
import numpy as np
import os

def __angle_between(p1, p2):  #두점 사이의 각도:(getAngle3P 계산용) 시계 방향으로 계산한다. P1-(0,0)-P2의 각도를 시계방향으로
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    res = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    return res



def getAngle3P(p1, p2, p3, direction="CW"): #세점 사이의 각도 1->2->3
    pt1 = (p1[0] - p2[0], p1[1] - p2[1])
    pt2 = (p3[0] - p2[0], p3[1] - p2[1])
    res = __angle_between(pt1, pt2)
    res = (res + 360) % 360
    if direction == "CCW":    #반시계방향
        res = (360 - res) % 360
    return res



def proc(module, video):
    angle = 0
    xy_1 =()
    xy_14 =()
    xy_8 =()
    
    
    if module is "coco":
        protoFile = "./model/coco.prototxt"
        weightsFile = "./model/coco.caffemodel"
        nPoints = 18
        POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

    elif module is "mpi":
        protoFile = "./model/mpi.prototxt"
        weightsFile = "./model/mpi.caffemodel"
        nPoints = 15
        POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
    inWidth = 368
    inHeight = 368
    threshold = 0.1
    input_source = "./input/" + video
    print(input_source)
    cap = cv2.VideoCapture(input_source)

    hasFrame, frame = cap.read()
    vid_writer = cv2.VideoWriter('./output/' + video.split(".")[0] + "_"+ module + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    while cv2.waitKey(1) < 0:
        t = time.time()
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        if not hasFrame:
            cv2.waitKey()
            break
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                  (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        H = output.shape[2]
        W = output.shape[3]
        points = []
        for i in range(nPoints):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H
            if prob > threshold : 
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                points.append((int(x), int(y)))
            else :
                points.append(None)
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, str(partA), points[partA], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                
                
            if partA == 14 and partB == 8:
                    xy_1 = points[partA]
                    xy_14 = points[partB]
                     
            elif partA == 8 and partB == 9:
                    xy_8 = points[partB]
                    
            if xy_1 != () and xy_8 != () and xy_14 != ():
                angle = getAngle3P(xy_8, xy_14, xy_1)
                cv2.putText(frame, str(round(angle,2)), xy_14, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
                xy_1 = ()
                xy_8 = ()
                xy_14 = ()
                if angle>100 and angle < 185:
                    cv2.putText(frame, "GOOD", (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
                else:
                    cv2.putText(frame, "BAD", (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
                
        #cv2.putText(frame, "Test Time = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        vid_writer.write(frame)
    vid_writer.release()

if __name__ == '__main__':
    module = ["mpi","coco"]
    for i in module:
        for j in os.listdir("./input"):
            if j[-3:] in ['mp4','avi']:
                proc(i, j)
