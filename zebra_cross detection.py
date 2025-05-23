import cv2
from ultralytics import YOLO

model = YOLO("crosswalk.pt")  
cap = cv2.VideoCapture(0)  

conf_threshold = 0.6 
iou_threshold = 0.45 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        source=frame,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',  
        verbose=False
    )
    
    if results[0].boxes:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"Crosswalk {conf:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Crosswalk Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()