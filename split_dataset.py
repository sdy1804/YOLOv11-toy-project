from glob import glob
from sklearn.model_selection import train_test_split

img_list = glob('/hdd/dongyeon/yolov11/Self_driving_dataset/custom_CCTV_dataset_highway_AIhub/export/images/*.png')
print(len(img_list))

train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=2000)

print(len(train_img_list), len(val_img_list))

with open('/hdd/dongyeon/yolov11/Self_driving_dataset/custom_CCTV_dataset_highway_AIhub/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open('/hdd/dongyeon/yolov11/Self_driving_dataset/custom_CCTV_dataset_highway_AIhub/val.txt', 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')

import yaml

with open('/hdd/dongyeon/yolov11/Self_driving_dataset/custom_CCTV_dataset_highway_AIhub/data3.yaml', 'r') as f:
    data = yaml.safe_load(f)

print(data)

data['train'] = '/hdd/dongyeon/yolov11/Self_driving_dataset/custom_CCTV_dataset_highway_AIhub/train.txt'
data['val'] = '/hdd/dongyeon/yolov11/Self_driving_dataset/custom_CCTV_dataset_highway_AIhub/val.txt'

with open('/hdd/dongyeon/yolov11/Self_driving_dataset/custom_CCTV_dataset_highway_AIhub/data3.yaml', 'w') as f:
    yaml.dump(data, f)

print(data)
