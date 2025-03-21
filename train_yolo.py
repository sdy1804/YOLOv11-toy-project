from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(data="/hdd/dongyeon/yolov11/Self_driving_dataset/custom_CCTV_dataset_highway_AIhub/data3.yaml", epochs=10, batch=8, imgsz=640, device=0)
