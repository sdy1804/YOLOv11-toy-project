import cv2
from ultralytics import solutions

cap = cv2.VideoCapture("/hdd/dongyeon/yolov11/highway_demo.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

region_points = [(160, 150), (400, 150), (400, 340), (160, 340)]

video_writer = cv2.VideoWriter("yolo_object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

counter = solutions.ObjectCounter(show=True, region=region_points, model="/hdd/dongyeon/yolov11/runs/detect/train6/weights/best.pt")

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = counter.count(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
