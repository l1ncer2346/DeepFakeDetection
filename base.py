import json
import os
import cv2
import math
base_path = '.\\train_sample_videos\\'

def GetFileName(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

def ReadFromJson():
    with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
        metadata = json.load(metadata_json)
        #print(len(metadata))
    return metadata
