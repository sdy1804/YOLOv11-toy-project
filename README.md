YOLOv11-toy-project
===================
  
  ![Image](https://github.com/user-attachments/assets/ae8d8427-bfaf-498c-8100-f2e1821ec229)

<!-- Failed to upload "count_car.gif" -->

### 1. introduction
This project is toy project using yolov11.   
Count object number in specific square area.
And it also provides tracklet visualization.

### 2. install
Python >= 3.8, Pytorch >= 1.8    
https://github.com/ultralytics/ultralytics?tab=readme-ov-file
```
pip install ultralytics
```
### 3. utilize codes
When using custom dataset to YOLOv11, split_dataset.py can split data to train and validation.    
trans_json_to_txt.py can transform format from .json to .txt that fits to yolo's.     
test_track.py is custom counting code. it used yolo's tracking function and custom counting algorithm.    
It counts objects whether yolo's bbox points(x1,y1,x2,y2) are in square area.    
yolo_object_count.py used yolo's counting system. It can be compared with ours.    
