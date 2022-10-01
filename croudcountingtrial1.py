import cv2
import numpy as np

cap= cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)


net =cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")

model= cv2.dnn.DetectionModel(net)
model.setInputParams(size=(320,320),scale=1/255)

#load class list
classes=[]
with open("dnn_model/classes.txt","r") as file_object:
  for class_name in file_object.readlines():
    class_name = class_name.strip()
    classes.append(class_name)
    


while True:
  _,frame = cap.read()
  height , width , _ = frame.shape
  frame[913:height -5, 89:89+1093]=(255,255,255)

#object detection
  
  (class_ids,score,bboxes)=model.detect(frame,nmsThreshold=0.4)
  i=0
  for class_id,score,bbox in zip(class_ids,score,bboxes):
    (x,y,w,h)= bbox
    class_name=classes[class_id]
    
    
    if class_name=="person":
      i=i+1
      
      cv2.rectangle(frame,(x,y),(x+w,y+h),(200,0,50),3)
  print(class_ids)
  polygon = np.array([[(20,20),(280,20),(280,70),(20,70)]])
  cv2.fillPoly(frame,polygon,(0,0,200))
  cv2.putText(frame,  "TOTAL PERSON : " + str(i), (30,60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
  

  
  cv2.imshow("camera", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()
