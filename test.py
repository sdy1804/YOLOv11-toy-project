from ultralytics import YOLO
import cv2

model = YOLO("/hdd/dongyeon/yolov11/yolo11n.pt")

# Run inference on an image
results = model.predict(source="bus.jpg", save=True, device=0)

# Plot inference results
plot = results[0].plot(font_size=0.1, pil=True)
