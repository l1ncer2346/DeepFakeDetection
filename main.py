import json
import os
import cv2
import math
import conventor as cvr
import base
import pandas

def main():
	conventor = cvr.ConventerFromVideoToImage(base.base_path)
	conventor.ConvertToImage()

if __name__ == '__main__':
	main()