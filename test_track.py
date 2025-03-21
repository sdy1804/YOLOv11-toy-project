from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("/hdd/dongyeon/yolov11/runs/detect/train6/weights/best.pt")
# model = YOLO("/hdd/dongyeon/yolov11/yolo11n.pt")

# Open the video file
input_video_path = "/hdd/dongyeon/yolov11/highway_demo.mp4"
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

track_history = defaultdict(lambda: [])

output_video_path = "/hdd/dongyeon/yolov11/runs/track/count_car.mp4"
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    pt1 = (160, 150)
    pt2 = (400, 340)
    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), thickness= 2)
    results = model.track(frame, persist= True, tracker='/hdd/dongyeon/yolov11/trackers/bytetrack.yaml')
    
    num_of_car = 0
    for result in results:
        annotated_frame = result.plot()
        boxes = result.boxes
        xyxy = boxes.xyxy
        xywh = boxes.xywh
        track_ids = []
        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
        for box, track_id in zip(xywh, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 0, 255), thickness=3)
        for xy in xyxy:
            if (xy[0] >= pt1[0] and xy[0] <= pt2[0] and  xy[1] >= pt1[1] and xy[1] <= pt2[1] or\
                    xy[2] >= pt1[0] and xy[2] <= pt2[0] and xy[3] >= pt1[0] and xy[3] <= pt2[1]):
                num_of_car += 1
    
    text = f"cars_in_area: {num_of_car}"
    cv2.putText(annotated_frame, text, (10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,255,0), thickness=2)
    out.write(annotated_frame)

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
