import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from vidgear.gears import CamGear

stream = CamGear(source='https://www.youtube.com/watch?v=EPKWu223XEg', stream_mode = True, logging=True).start() # YouTube Video URL as input

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)



my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

area = [(588,365), (512,437), (868,790), (993,698)]


count=0
while True:    
    frame = stream.read()
    count += 1
    if count % 3 != 0:
        continue
   
   
    frame = cv2.resize(frame, (1020, 800))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list1=[]
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        if 'car' in c:
           result=cv2.pointPolygonTest(np.array(area, np.int32),((cx,cy)),False)
           if result>=0:
              cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
              list1.append(cy)
            

    carcounter=len(list1)
    total=6-carcounter
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    cvzone.putTextRect(frame,f'freespace:-{total}',(50,60),2,2)
    cvzone.putTextRect(frame,f'carcounter:-{carcounter}',(50,160),2,2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

stream.stop()   
cv2.destroyAllWindows()
