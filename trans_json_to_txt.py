from tqdm import tqdm
from pathlib import Path
import os
import sys
import json
import shutil
import numpy as np


LABEL = {
    "안전모": 0,
    "안전대": 1,
    "마스크": 2,
    "장갑": 3,
    "보안경": 4,
}

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

def transform_yolo_coord(coords:list) -> list:
    """
    입력된 절대 좌표를 YOLO 형식의 상대 좌표로 변환하는 함수.
    
    Args:
    coords (list): [[x_min, y_min], [x_max, y_max]] 형태의 절대 좌표 리스트.
    
    Returns:
    list: YOLO 형식의 좌표 [x_center, y_center, width, height]
    """
    
    # 절대 좌표에서 x_min, y_min, x_max, y_max 추출
    x_min, y_min, x_max, y_max = coords
    
    # 중심 좌표 계산
    x_center = (x_min + x_max) / 2 / IMAGE_WIDTH
    y_center = (y_min + y_max) / 2 / IMAGE_HEIGHT
    
    # 너비와 높이 계산
    width = (x_max - x_min) / IMAGE_WIDTH
    height = (y_max - y_min) / IMAGE_HEIGHT
    
    # YOLO 형식으로 반환
    return ' '.join(map(str, np.round(np.array([x_center, y_center, width, height]), decimals=6).tolist()))


def json_to_txt(json_file:Path, save_path:Path) -> None:
    ...


if __name__ == "__main__":
    code_path, input_path = map(Path, sys.argv)
    output_path = Path(input_path.parent, "dataset")

    train_split_portion = 0.7
    valid_split_portion = 0.2
    test_split_portion = 0.1


    src_files = {f.stem:f for f in input_path.rglob("*.jpg")}
    json_files = {f.stem:f for f in input_path.rglob("*.json")}

    train_point = int(len(src_files)*train_split_portion)
    valid_point = int(len(src_files)*valid_split_portion)

    # lables
    
    save_train_json  = Path(output_path, "labels", "train")
    save_val_json  = Path(output_path, "labels", "val")
    save_test_json  = Path(output_path, "labels", "test")
    os.makedirs(save_train_json, exist_ok=True)
    os.makedirs(save_val_json, exist_ok=True)
    os.makedirs(save_test_json, exist_ok=True)

    # val
    save_train_src = Path(output_path, "images", "train")
    save_val_src = Path(output_path, "images", "val")
    save_test_src = Path(output_path, "images", "test")
    os.makedirs(save_train_src, exist_ok=True)
    os.makedirs(save_val_src, exist_ok=True)
    os.makedirs(save_test_src, exist_ok=True)
    

    # train
    for key in tqdm(list(src_files.keys())[:train_point], desc="(1/3) processing / train"):
        json_file = json_files[key]
        src_file = src_files[key]
        
        # trnsform json
        json_to_txt(json_file, save_train_json)

        # moving src
        if not os.path.isfile(str(save_train_src / src_file.name)):
            shutil.copyfile(str(src_file), str(save_train_src / src_file.name))

    # valid
    for key in tqdm(list(src_files.keys())[train_point:train_point+valid_point], desc="(2/3) processing / val"):
        json_file = json_files[key]
        src_file = src_files[key]
        
        # trnsform json
        json_to_txt(json_file, save_val_json)

        # moving src
        if not os.path.isfile(str(save_val_src / src_file.name)):        
            shutil.copyfile(str(src_file), str(save_val_src / src_file.name))

    # test
    for key in tqdm(list(src_files.keys())[train_point+valid_point:], desc="(3/3) processing / test"):
        json_file = json_files[key]
        src_file = src_files[key]
        
        # trnsform json
        json_to_txt(json_file, save_test_json)

        # moving src
        if not os.path.isfile(str(save_test_src / src_file.name)):        
            shutil.copyfile(str(src_file), str(save_test_src / src_file.name))

    print("작업 완료")
